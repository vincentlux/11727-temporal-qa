from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, MultiMarginLoss, BCEWithLogitsLoss
import torch.nn.functional as F
import torch.nn.init as init
import math
from transformers import BertPreTrainedModel,RobertaConfig, RobertaModel,ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, BertModel
from transformers import AlbertModel

class Trilinear_Att_layer(nn.Module):
    def __init__(self, config):
        super(Trilinear_Att_layer, self).__init__()
        self.W1 = nn.Linear(config.hidden_size, 1)
        self.W2 = nn.Linear(config.hidden_size, 1) 
        self.W3 = nn.Parameter(torch.Tensor(1, 1, config.hidden_size))
        init.kaiming_uniform_(self.W3, a=math.sqrt(5))

    def forward(self, u, u_mask, v, v_mask):
        part1 = self.W1(u)     # batch * seq_len * 1
        part2 = self.W2(v).permute(0, 2, 1)   # batch * 1 * seq_len
        part3 = torch.bmm(self.W3*u, v.permute(0, 2, 1))  # batch * seq_len * seq_len
        u_mask = (1.0 - u_mask.float()) * -10000.0
        v_mask = (1.0 - v_mask.float()) * -10000.0
        joint_mask = u_mask.unsqueeze(2) + v_mask.unsqueeze(1)    # batch * seq_len * num_paths
        total_part = part1 + part2 + part3 + joint_mask
        return total_part

class OCN_Att_layer(nn.Module):
    def __init__(self, config):
        super(OCN_Att_layer, self).__init__()
        self.att = Trilinear_Att_layer(config)

    def forward(self, ol, ol_mask, ok, ok_mask):
        A = self.att(ol, ol_mask, ok, ok_mask)
        att = F.softmax(A, dim=1)    
        _OLK = torch.bmm(ol.permute(0, 2, 1), att).permute(0, 2, 1)       # batch *  hidden * seq_len
        OLK = torch.cat([ok-_OLK, ok*_OLK], dim=2)
        return OLK

class OCN_CoAtt_layer(nn.Module):
    def __init__(self, config):
        super(OCN_CoAtt_layer, self).__init__()
        self.att = Trilinear_Att_layer(config)
        self.Wp = nn.Linear(config.hidden_size*3, config.hidden_size)

    def forward(self, d, d_mask, OCK, OCK_mask):
        A = self.att(d, d_mask, OCK, OCK_mask)
        ACK = F.softmax(A, dim=2)    
        OA = torch.bmm(ACK, OCK)   
        APK = F.softmax(A, dim=1)
        POAA = torch.bmm(torch.cat([d, OA], dim=2).permute(0, 2, 1), APK).permute(0, 2, 1)
        OPK = F.relu(self.Wp(torch.cat([OCK, POAA], dim=2)))
        return OPK

class OCN_SelfAtt_layer(nn.Module):
    def __init__(self, config):
        super(OCN_SelfAtt_layer, self).__init__()
        self.att = Trilinear_Att_layer(config)
        self.Wf = nn.Linear(config.hidden_size*4, config.hidden_size)

    def forward(self, OPK, OPK_mask, _OPK, _OPK_mask):
        A = self.att(OPK, OPK_mask, _OPK, _OPK_mask)
        att = F.softmax(A, dim=1)    
        OSK = torch.bmm(OPK.permute(0, 2, 1), att).permute(0, 2, 1)      
        OFK = torch.cat([_OPK, OSK, _OPK-OSK, _OPK*OSK], dim=2)
        OFK = F.relu(self.Wf(OFK))
        return OFK

class OCN_Merge_layer(nn.Module):
    def __init__(self, config):
        super(OCN_Merge_layer, self).__init__()
        self.Wc_self = nn.Linear(config.hidden_size, config.hidden_size)
        self.Wc = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.Va = nn.Linear(config.hidden_size, 1)
        self.Wg = nn.Linear(config.hidden_size*3, config.hidden_size)

    def forward(self, o1, compare_list, q, q_mask):
        q_mask = (1.0 - q_mask.float()) * -10000.0
        Aq = F.softmax(self.Va(q)+q_mask.unsqueeze(2), dim=1)
        Q = torch.bmm(q.permute(0, 2, 1), Aq).permute(0, 2, 1).repeat(1, o1.shape[1], 1)
        #OCK = torch.tanh(self.Wc(torch.cat([o1, o1o2, o1o3, o1o4, o1o5], dim=2)))   # batch * seq_len * hidden
        o_proj = [self.Wc_self(o1).unsqueeze(2)]+[self.Wc(o).unsqueeze(2) for o in compare_list]
        OCK = torch.tanh(torch.sum(torch.cat(o_proj, dim=2), dim=2))
        G = torch.sigmoid(self.Wg(torch.cat([o1, OCK, Q], dim=2)))
        out = G*o1 + (1-G)*OCK
        return out

class OptionCompareCell(nn.Module):

    def __init__(self, config):
        super(OptionCompareCell, self).__init__()
        self.option_att_layer = OCN_Att_layer(config)
        self.option_merge_layer = OCN_Merge_layer(config)
        self.CoAtt_layer = OCN_CoAtt_layer(config)
        self.SelfAtt_layer = OCN_SelfAtt_layer(config)

    def forward(self, encoded_o, encoded_q, option_mask, question_mask):
        num_cand = encoded_o.size(1)
        final_options = []
        for i in range(num_cand):
            compare = []
            for j in range(num_cand):
                if i != j:
                    oioj = self.option_att_layer(encoded_o[:, j, :, :], option_mask[:, j, :], encoded_o[:, i, :, :], option_mask[:, i, :])
                    compare.append(oioj)
            merged_oi = self.option_merge_layer(encoded_o[:, i, :, :], compare, encoded_q[:, i, :, :], question_mask[:, i, :])
            reread_oi = self.CoAtt_layer(encoded_q[:, i, :, :], question_mask[:, i, :], merged_oi, option_mask[:, i, :])
            final_oi = self.SelfAtt_layer(reread_oi, option_mask[:, i, :], reread_oi, option_mask[:, i, :])
            final_options.append(final_oi)
        candidates = torch.cat([final_o.unsqueeze(1) for final_o in final_options], dim=1)
        candidates, _ = torch.max(candidates, dim=2)
        return candidates

class RobertaForPIQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(RobertaForPIQA, self).__init__(config)
        self.num_labels = config.num_labels    
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def _resize_type_embeddings(self, new_num_types):
        old_embeddings = self.roberta.embeddings.token_type_embeddings
        new_embeddings = self.roberta._get_resized_embeddings(old_embeddings, new_num_types)
        self.roberta.embeddings.token_type_embeddings = new_embeddings
        return self.roberta.embeddings.token_type_embeddings

    def forward(self, task=None, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        seq_length = input_ids.size(2)
        outputs = self.roberta(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length), 
            token_type_ids=token_type_ids.view(-1, seq_length),
            position_ids=position_ids, 
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
       
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        # import pdb; pdb.set_trace()

        logits = logits.view(-1, self.num_labels)
        outputs = (logits,) + outputs[2:] 

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, out_size=1):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, out_size)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class RobertaForMCTACO(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(RobertaForMCTACO, self).__init__(config)
        self.num_labels = config.num_labels    
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config,out_size=2)
        self.classifier_bce = RobertaClassificationHead(config,out_size=1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def _resize_type_embeddings(self, new_num_types):
        old_embeddings = self.roberta.embeddings.token_type_embeddings
        new_embeddings = self.roberta._get_resized_embeddings(old_embeddings, new_num_types)
        self.roberta.embeddings.token_type_embeddings = new_embeddings
        return self.roberta.embeddings.token_type_embeddings

    def forward(self, task=None, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        seq_length = input_ids.size(2)
        outputs = self.roberta(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length), 
            token_type_ids=token_type_ids.view(-1, seq_length),
            position_ids=position_ids, 
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
       
        # import pdb; pdb.set_trace()
        sequence_output = outputs[0]
        

        if task == 'mctaco-bce':
            logits = self.classifier_bce(sequence_output)
            logits = logits.view(-1)
            logits_return = self.sigmoid(logits)
            
            outputs = (logits_return,) + outputs[2:] 
            if labels is not None:
                # use BCEWithLogitsLoss to avoid precision bug
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.view(-1).float())
                outputs = (loss,) + outputs
        else:
            logits = self.classifier(sequence_output)
            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)




class BertForMCTACO(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.OCN_cell = OptionCompareCell(config)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.hidden_size = config.hidden_size
        self.loss_fct = CrossEntropyLoss()
        self.init_weights()

    def forward(self, task=None, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
        head_mask=None, inputs_embeds=None, labels=None,):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        # )
        batch_size, num_cand, seq_length = input_ids.shape
        outputs = self.bert(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length), 
            token_type_ids=token_type_ids.view(-1, seq_length),
            position_ids=position_ids, 
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        all_hidden = outputs[0]
        option_mask = ((attention_mask == 1) & (token_type_ids == 1)).long()
        question_mask = ((attention_mask == 1) & (token_type_ids == 0)).long()
        encoded = all_hidden.view(batch_size, num_cand, seq_length, self.hidden_size)
        encoded_o = encoded * option_mask.unsqueeze(3).float()
        encoded_q = encoded * question_mask.unsqueeze(3).float()
        candidates = self.OCN_cell(encoded_o, encoded_q, option_mask, question_mask)
        #pooled_output = outputs[1]

        #pooled_output = self.dropout(pooled_output)
        
        # if task == 'mctaco-bce':
        #     logits = self.classifier_bce(pooled_output)
        #     logits = logits.view(-1)
        #     logits_return = self.sigmoid(logits)
            
        #     outputs = (logits_return,) + outputs[2:] 
        #     if labels is not None:
        #         # use BCEWithLogitsLoss to avoid precision bug
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(logits, labels.view(-1).float())
        #         outputs = (loss,) + outputs
        # else:
        logits = self.classifier(candidates)
        logits = logits.view(-1, self.num_labels)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

