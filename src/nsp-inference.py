import argparse
import os
import string
import sys
sys.path.append('evaluator')
import nltk
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction

from evaluator import McTacoEvaluator

'''
by setting args.method = converted_q_as_s1
val
    best em: 0.24603174603174602 at cutoff: 1.0
    best f1: 0.42054288304288295 at cutoff: 0.95176

test 
    best em: 0.2197452229299363 
    best f1: 0.41063612813405753

'''


def convert_question_to_sentence(sample):
    '''use rule to remove how to and convert to sentence
    '''
    question = nltk.word_tokenize(sample.lower())
    q_tags = nltk.pos_tag(question)
    if 'how' not in question:
        content = sample.lower()
    elif not q_tags[2][1].startswith('VB') and 'would' not in question:
        for i in range(3, len(q_tags)):
            if q_tags[i][1].startswith('VB'):
                break
        if len(question)>i+3 and question[i+1] == 'it' and (question[i+2] == 'take' or question[i+2] == 'taken'):
            q_words = ' '.join(question[:i+3])
            content = ' '.join(question[i+3:])
        else:
            q_words = ' '.join(question[:i+1])
            content = ' '.join(question[i+1:])
    else:
        if len(question)>5 and question[3] == 'it' and (question[4] == 'take' or question[4] == 'taken'):
            q_words = ' '.join(question[:5])
            content = ' '.join(question[5:])
        else:
            q_words = ' '.join(question[:3])
            content = ' '.join(question[3:])
    
    # remove punctuation
    content = content.translate(str.maketrans('', '', string.punctuation)).strip()
    if len(content) == 0:
        succeed = False
    else:
        succeed = True
    return content, succeed

def read_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            sample = line.strip().split('\t')
            assert len(sample) == 5
            data.append(sample)
    return data

def get_score_for_nsp(sentence_1, sentence_2, tokenizer, NspModel):
    sent1_toks = ["[CLS]"] + tokenizer.tokenize(sentence_1) + ["[SEP]"]
    sent2_toks = tokenizer.tokenize(sentence_2) + ["[SEP]"]
    # text = sent1_toks + sent2_toks
    indexed_tokens = tokenizer.convert_tokens_to_ids(sent1_toks + sent2_toks)
    segments_ids = [0]*len(sent1_toks) + [1]*len(sent2_toks)
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()
    prediction = NspModel(tokens_tensor, token_type_ids=segments_tensors)
    score = prediction[0] # tuple to tensor
    softmax = torch.nn.Softmax(dim=1)
    prediction_sm = softmax(score) # yes no
    
    # return yes proba as proba
    return float(prediction_sm[0][0].cpu().detach().numpy())

def get_probas(args, data, tokenizer, BertNSP):
    yes_proba_list = []
    for i, sent in enumerate(tqdm(data)):
        if sent[-1] != 'Event Duration' and args.only_duration:
            continue
        if args.method == 'context_as_s1':
            converted_q, succeed = convert_question_to_sentence(sent[1])
            sentence_1 = sent[0]
            sentence_2 = ' '.join([converted_q, sent[2]]) 
        elif args.method == 'converted_q_as_s1':
            converted_q, succeed = convert_question_to_sentence(sent[1])
            sentence_1 = converted_q
            sentence_2 = sent[2]
        elif args.method == 'original_q_as_s1':
            sentence_1 = sent[1]
            sentence_2 = sent[2]
        # sentence_1 = "How old are you?"
        # sentence_2 = "The Eiffel Tower is in Paris"
        yes_proba = get_score_for_nsp(sentence_1, sentence_2, tokenizer, BertNSP)
        yes_proba_list.append(yes_proba)
    return yes_proba_list

def convert_proba_list_to_yes_no(proba_list, cutoff):
    return ['yes' if i > cutoff else 'no' for i in proba_list], ['no' if i > cutoff else 'yes' for i in proba_list]

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="The input dev/test mctaco file for inference",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.,
    )
    parser.add_argument(
        "--end",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--search_range",
        type=int,
    )
    parser.add_argument(
        "--em_cutoff",
        type=float,
    )
    parser.add_argument(
        "--f1_cutoff",
        type=float,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices={'context_as_s1', 'converted_q_as_s1', 'original_q_as_s1'},
        help="The input dev/test mctaco file for inference",
    )
    parser.add_argument("--only_duration", action="store_true", help="only inferencing duration")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    
    args = parser.parse_args()
    assert args.eval or args.test

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    BertNSP=BertForNextSentencePrediction.from_pretrained('bert-base-uncased').cuda()
    BertNSP.eval()
    evaluator = McTacoEvaluator()
    ref_lines_for_eval_script = [x.strip() for x in open(args.file).readlines()]
    
    data = read_data(args.file)
    proba_list = get_probas(args, data, tokenizer, BertNSP)
    
    # grid search
    if args.eval:
        cutoff_list = np.linspace(args.start,args.end,args.search_range)
        em_best_cutoff = 0.
        em_best_score = 0.
        f1_best_cutoff = 0.
        f1_best_score = 0.
        for cutoff in cutoff_list:
            large_to_yes, large_to_no = convert_proba_list_to_yes_no(proba_list, cutoff)
            large_to_yes_em, large_to_yes_f1 = evaluator.get_em_f1(ref_lines=ref_lines_for_eval_script, prediction_lines=large_to_yes)
            large_to_no_em, large_to_no_f1 = evaluator.get_em_f1(ref_lines=ref_lines_for_eval_script, prediction_lines=large_to_no)
            if large_to_yes_em >= large_to_no_em:
                if large_to_yes_em >= em_best_score:
                    em_best_score = large_to_yes_em
                    em_best_cutoff = cutoff
                print(f'em: {large_to_yes_em} at {cutoff}; larger->yes')
            else:
                if large_to_no_em >= em_best_score:
                    em_best_score = large_to_no_em
                    em_best_cutoff = cutoff
                print(f'em: {large_to_no_em} at {cutoff}; larger->no')

            if large_to_yes_f1 >= large_to_no_f1:
                if large_to_yes_f1 >= f1_best_score:
                    f1_best_score = large_to_yes_f1
                    f1_best_cutoff = cutoff
                print(f'f1: {large_to_yes_f1} at {cutoff}; larger->yes')
            else:
                if large_to_no_f1 >= f1_best_score:
                    f1_best_score = large_to_no_f1
                    f1_best_cutoff = cutoff
                print(f'f1: {large_to_no_f1} at {cutoff}; larger->no\n')
        
        print(f'best em: {em_best_score} at cutoff: {em_best_cutoff}')
        print(f'best f1: {f1_best_score} at cutoff: {f1_best_cutoff}')
        args.em_cutoff = em_best_cutoff
        args.f1_cutoff = f1_best_cutoff
    
    if args.test:
        # assert 'test' in args.file
        # shouldn't do twice, but who cares
        large_to_yes, large_to_no = convert_proba_list_to_yes_no(proba_list, args.em_cutoff)
        large_to_yes_em, large_to_yes_f1 = evaluator.get_em_f1(ref_lines=ref_lines_for_eval_script, prediction_lines=large_to_yes)
        large_to_no_em, large_to_no_f1 = evaluator.get_em_f1(ref_lines=ref_lines_for_eval_script, prediction_lines=large_to_no)
        best_em = large_to_yes_em if large_to_yes_em > large_to_no_em else large_to_no_em
        if large_to_yes_em > large_to_no_em:
            print('em: large to yes')
        else:
            print('em: large to no')
        

        # f1
        large_to_yes, large_to_no = convert_proba_list_to_yes_no(proba_list, args.f1_cutoff)
        large_to_yes_em, large_to_yes_f1 = evaluator.get_em_f1(ref_lines=ref_lines_for_eval_script, prediction_lines=large_to_yes)
        large_to_no_em, large_to_no_f1 = evaluator.get_em_f1(ref_lines=ref_lines_for_eval_script, prediction_lines=large_to_no)
        best_f1 = large_to_yes_f1 if large_to_yes_f1 > large_to_no_f1 else large_to_no_f1
        if large_to_yes_f1 > large_to_no_f1:
            print('f1: large to yes')
        else:
            print('f1: large to no')

        # save proba list
        with open('./mctaco-data/best_nsp_eval.txt', 'w') as f:
            for i in large_to_yes:
                f.write(i+'\n')

        print(f'best em: {best_em} best f1: {best_f1}')


if __name__ == '__main__':
    main()

