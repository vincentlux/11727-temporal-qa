import os
import json
import csv
from word2number import w2n    
import nltk
from nltk.stem.wordnet import WordNetLemmatizer



class TimeNormalize:
    def __init__(self):
        self.lmtzr = WordNetLemmatizer()
        self.unit_map = ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years', 'decades', 'centuries']
        
        self.num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
             6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
            11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
            15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
            19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
            50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
            90: 'Ninety', 0: 'Zero'}
        
        # self.convert_map = {
        #     "seconds": 1.0,
        #     "minutes": 60.0,
        #     "hours": 60.0 * 60.0,
        #     "days": 24.0 * 60.0 * 60.0,
        #     "weeks": 7.0 * 24.0 * 60.0 * 60.0,
        #     "months": 30.0 * 24.0 * 60.0 * 60.0,
        #     "years": 365.0 * 24.0 * 60.0 * 60.0,
        #     "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
        #     "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        # }
        self.convert_map = {
            "seconds": 0.0,
            "minutes": 10.0,
            "hours": 20.0,
            "days": 30.0,
            "weeks": 40.0,
            "months": 50.0,
            "years": 60.0,
            "decades": 70.0,
            "centuries": 80.0,
        }
        self.unit_div = {
            "seconds": 6,
            "minutes": 6,
            "hours": 2.4,
            "days": 0.7,
            "weeks": 0.4,
            "months": 1.2,
            "years": 1,
            "decades": 1,
            "centuries": 1,
        }
        self.lemmatized_convert_map = {self.lmtzr.lemmatize(k): v for k, v in self.convert_map.items()}
        self.lemmatized_unit_div = {self.lmtzr.lemmatize(k): v for k, v in self.unit_div.items()}

    def convert_time_to_seconds(self, cand):
        """if time in a string detected convert to and return number in seconds
        Args:
            cand (str)

        Returns:
            bool: representing if successfully converted 
            float: representing equivalent seconds        
        """
        cand = cand.replace('night', 'day')
        cand = cand.replace('morning', 'day')
        cand = cand.replace('season', 'month')
        # if already only number, return this number
        if self.get_trivial_floats(cand) is not None:
            #return True, self.get_trivial_floats(cand)
            return True, self.get_trivial_floats(cand)/6.0

        # if length is one and already unit, return in seconds
        check_length = len(cand.split(' '))
        if check_length == 1 and self.lmtzr.lemmatize(cand) in self.lemmatized_convert_map:
            return True, self.lemmatized_convert_map[self.lmtzr.lemmatize(cand)]

        split = cand.rsplit(' ', 1)      # split from right by one
        number, unit = split[0], split[-1]
        quantized_number = self.quantity([number], convert_surface_words_to_floats=True, convert_float_to_int=False)
        # if failed to extract, try last one and last two of number (hardcode)
        if not quantized_number:
            quantized_number = self.quantity([number.split(' ')[-1]], convert_surface_words_to_floats=True, convert_float_to_int=False)
        # if not quantized_number:
        #     quantized_number = self.quantity([number.split(' ')[-1]], convert_surface_words_to_floats=True, convert_float_to_int=False)

        # lemmatize unit and get number of seconds from map
        if quantized_number:
            lemmatized_unit = self.lmtzr.lemmatize(unit)
            if lemmatized_unit in self.lemmatized_convert_map:
                #return True, quantized_number * self.lemmatized_convert_map[lemmatized_unit]
                return True, quantized_number/self.lemmatized_unit_div[lemmatized_unit]+self.lemmatized_convert_map[lemmatized_unit]
            else:
                return False, None
        else:
            # if cannot get number but get valid unit, just use unit
            lemmatized_unit = self.lmtzr.lemmatize(unit)
            if lemmatized_unit in self.lemmatized_convert_map:
                return True, self.lemmatized_convert_map[lemmatized_unit]
            else:
                return False, None        



    def get_normalized_field(self, cand, convert_surface_words_to_floats=False, number_included=False, post_process_century=True, round_method='down'):            
        check_length = len(cand.split(' '))
        if check_length >= 3:
            print([cand])
            return cand      
        split = cand.rsplit(' ', 1)      # split from right by one
        number, unit = split[0], split[-1]
        quantized_number = self.quantity([number], convert_surface_words_to_floats)
        if not quantized_number or len(split) <= 1:
            # print(f'cannot quantity {split[0]} in {cand}')
            return cand
        
        quantized_expression = ' '.join([str(quantized_number), unit])
        if unit in self.unit_map:
            converted_time, converted_unit = self.normalize_timex(quantized_expression, post_process_century, round_method)
            if not number_included:
                return converted_unit
            res = ' '.join([str(converted_time), str(converted_unit)])
            return res
        else:
            # if failed to normalize unit return the original expression with number quantized
            # also int() to number
            quantized_expression = ' '.join([str(int(quantized_number)), unit])
            return quantized_expression


    def post_convert_num_to_word(self, str_in, only_convert_first=True, not_convert_time=True):

        words = str_in.split(' ')
        for i, word in enumerate(words):
            if i > 0 and only_convert_first:
                return ' '.join([j for j in words])
            try:
                if int(word) in self.num2words.keys():
                    if not_convert_time:
                        if i < (len(words) - 1) and words[i+1] in ['am', 'pm', 'a.m', 'p.m', 'a.m.', 'p.m.']:
                            continue
                    words[i] = self.num2words[int(word)].lower()
            except:
                continue
        return ' '.join([i for i in words])



    
    def get_trivial_floats(self, s):
        try:
            n = float(s)
            return n
        except:
            return None

    # Extracting the surface numerical value amoung tokens
    def get_surface_floats(self, tokens):
        if tokens[-1] in ["a", "an"]:
            return 1.0
        if tokens[-1] == "several":
            return 4.0
        if tokens[-1] == "many":
            return 10.0
        if tokens[-1] == "some":
            return 3.0
        if tokens[-1] == "few":
            return 3.0
        if tokens[-1] == "tens" or " ".join(tokens[-2:]) == "tens of":
            return 10.0
        if tokens[-1] == "hundreds" or " ".join(tokens[-2:]) == "hundreds of":
            return 100.0
        if tokens[-1] == "thousands" or " ".join(tokens[-2:]) == "thousands of":
            return 1000.0
        if " ".join(tokens[-2:]) in ["a few", "a couple"]:
            return 3.0
        if " ".join(tokens[-3:]) == "a couple of":
            return 2.0
        return None

    # Extracting comprehensive numerical values
    def quantity(self, tokens, convert_surface_words_to_floats=False, convert_float_to_int=True):
        try:
            if self.get_trivial_floats(tokens[-1]) is not None:
                if convert_float_to_int:
                    return int(self.get_trivial_floats(tokens[-1]))
                else:
                    return self.get_trivial_floats(tokens[-1])
            
            if self.get_surface_floats(tokens) is not None:
                if convert_surface_words_to_floats:
                    return self.get_surface_floats(tokens)
                else:
                    return None
            string_comb = tokens[-1]
            cur = w2n.word_to_num(string_comb)
            for i in range(-2, max(-(len(tokens)) - 1, -6), -1):
                status = True
                try:
                    _ = w2n.word_to_num(tokens[i])
                except:
                    status = False
                if tokens[i] in ["-", "and"] or status:
                    if tokens[i] != "-":
                        string_comb = tokens[i] + " " + string_comb
                    update = w2n.word_to_num(string_comb)
                    if update is not None:
                        cur = update
                else:
                    break
            if cur is not None:
                return float(cur)
        except Exception as e:
            return None   

    # Normalizing a temporal expression to the nearest unit
    def normalize_timex(self, expression, post_process_century=True, round_method='down'):
        u = expression.split()[1]
        v_input = float(expression.split()[0])

        if u in ["instantaneous", "forever"]:
            return u, str(1)

        seconds = self.convert_map[u] * float(v_input)
        prev_unit = "seconds"
        for i, v in enumerate(self.convert_map):
            if seconds / self.convert_map[v] <= 1.0:
                break
            
            prev_unit = v
        if prev_unit == "seconds" and seconds > 60.0:
            prev_unit = "centuries"

        converted_time = seconds / self.convert_map[prev_unit]
        if post_process_century:
            # NOTE specifically change century behavior
            if prev_unit == "centuries": 
                if 10.0 < converted_time < 100.0:
                    prev_unit = "tens of centuries"
                elif 100.0 <= converted_time < 1000.0:
                    prev_unit = "hundreds of centuries"
                elif converted_time > 1000.0:
                    prev_unit = "thousands of centuries"
                else:
                    pass
                # set converted time to empty since time already included in the prev_unit
                converted_time = converted_time if prev_unit == "centuries" else ''
        if converted_time:
            if round_method == 'down':
                converted_time = int(converted_time)
            elif round_method == 'closest':
                converted_time = int(round(converted_time))

        return converted_time, prev_unit

