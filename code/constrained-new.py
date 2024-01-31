
#  need to run following 4
#!pip install openai transformers
#!pip install torch 
# !pip install sentencepiece
# !pip install protobuf==3.20.1


# !conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
# !conda install ipywidgets -c conda-forge



import pandas as pd
import numpy as np



# import sentencepiece

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

# rinna GPT-2

# tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")

# model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium").cuda()

# # gpt-2 medium

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

# model = AutoModelForCausalLM.from_pretrained("gpt2-medium").cuda()

# # mGPT

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")

# model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT").cuda()

# # NeoX

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").cuda()

# # NeoX-Japanese

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")

# model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b").cuda()

# # BERT m

# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased").cuda()

# # BERT Japanese

# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

# model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese").cuda()

#from google.colab import drive
#drive.mount('/content/drive')

# GPT

men_pmpt = pd.read_csv('men_gpt_prompt_new.csv')
wom_pmpt = pd.read_csv('wom_gpt_prompt_new.csv')


# occupations = pd.read_csv('occupations_2020.csv')  # to 1920
# occ = occupations['職業']

# remove duplicates and drop certain occ in occupations df

# occupations = pd.read_csv('小分類_1920_T9.csv')  # to 1920
# #occ = occupations['職業']
# print(len(occupations.index))
# occupations = occupations[occupations['職業'].str.contains('分類不能') == False]
# occupations = occupations[occupations['職業'].str.contains('その他') == False]
# occupations.drop_duplicates(subset=['職業'])
# print(len(occupations.index))

# occupations.to_csv('occupations_1920.csv', index = False)

# occupations.to_csv('occupations_2020.csv', index = False)

# def get_score(prompt, job):
#   input = tokenizer(prompt, return_tensors="pt").to("cuda")
#   input_ids = input.input_ids

#   force = tokenizer([job], add_special_tokens=False).input_ids
#  # print(force)
#   outputs = model.generate(
#       input_ids,
#       force_words_ids=force,
#       num_beams=5,
#       max_new_tokens = 5,
#       return_dict_in_generate=True,
#       output_scores=True
#       )

#   transition_scores = model.compute_transition_scores(
#       outputs.sequences, outputs.scores, normalize_logits=True
#   )

#   input_length = input_ids.shape[1]
#   generated_tokens = outputs.sequences[:, input_length:]
#   print(transition_scores)

#   true_score = -9999

#   for tok, score in zip(generated_tokens[0], transition_scores[0]):
#       # | token | token string | logits | probability
#       # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score:.3f} | ")
#       if tokenizer.decode(tok) == job:
#         true_score = score

#  # if force in 

#   # input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
#   # generated_tokens = outputs.sequences[:, input_length:]
#   # logit = transition_scores[0][0].cpu().item()  # logit of first token

#   # force_index = generated_tokens.index(force[0].item())
#   # logit = transition_scores[force_index]

#   return true_score.cpu().item()

# def get_score(prompt, job):
#   input = tokenizer(prompt, return_tensors="pt").to("cuda")
#   input_ids = input.input_ids

#   force = tokenizer([job], add_special_tokens=False).input_ids
#   #print(force[0][1]) #.cpu())
#   outputs = model.generate(
#       input_ids,
#       force_words_ids=force,
#       num_beams=5,
#       max_new_tokens = 10,
#       # num_return_sequences = 1,
#       return_dict_in_generate=True,
#       output_scores=True
#       )

#   transition_scores = model.compute_transition_scores(
#       outputs.sequences, outputs.scores, normalize_logits=True
#   )

#   input_length = input_ids.shape[1]
#   # print(input_length)
#   generated_tokens = outputs.sequences[:, input_length:]
#   # print(transition_scores)

#   # # print(generated_tokens[0]) #.cpu())
#   #for tok, score in zip(generated_tokens[0], transition_scores[0]):
#       #print(tokenizer.decode(tok))
#       # if tokenizer.decode(tok) == job:
#       #   true_score = score

# #   t = torch.Tensor([1, 2, 3])
# # print ((t == 2).nonzero(as_tuple=True)[0])

#   t = generated_tokens[0]
#   # print(t)
#   # print(force[0][1])

#   try:
#     index = ((t == force[0][1]).nonzero(as_tuple=True)[0])[0].item() 
#   except:
#     return -100

  
#   # print(tokenizer.decode(generated_tokens[0][index].item()))

#   # index =  force[0][1] # force.cpu()
#   # # index = generated_tokens[0].item()
#   # # print(index)

#   # if true_score == -9999:
#   #   return true_score

#   # return true_score.cpu().item()

#   return transition_scores[0][index].cpu().item() #.[0]][index].cpu().item()

def get_score(prompt, job, model, tokenizer):
  input = tokenizer(prompt, return_tensors="pt").to("cuda")
  input_ids = input.input_ids

  force = tokenizer([job], add_special_tokens=False).input_ids
  outputs = model.generate(
      input_ids,
      force_words_ids=force,
      num_beams=5,
      max_new_tokens = 10,
      return_dict_in_generate=True,
      output_scores=True
      )

  transition_scores = model.compute_transition_scores(
      outputs.sequences, outputs.scores, normalize_logits=True
  )

  input_length = input_ids.shape[1]
  generated_tokens = outputs.sequences[:, input_length:]

  t = generated_tokens[0]

  try:
    index = ((t == force[0][1]).nonzero(as_tuple=True)[0])[0].item() 
  except:
    return -100

  return transition_scores[0][index].cpu().item() #.[0]][index].cpu().item()

# test get_score

# a = get_score('太郎の仕事は', '医者')
# # b = get_score('太郎の仕事は', '助産師')
# # c = get_score('太郎の仕事は', '保育')

# print(a)
# # print(b)
# # print(c)

# tensor(-11.7443, device='cuda:0')
# tensor(-11.9802, device='cuda:0')
# tensor(-6.8715, device='cuda:0')

def get_distribution(prompt, occ, model, tokenizer):
    scores = [] # keeps max of each row
    raw = [] # keeps all occupations
    for o in occ:
      # print(o)

      # cur_score = get_score(prompt, o)

      cur_score = -9999

      if '(' in o:
        o = o.split('(')[0]
      if '従事者' in o:
        o = o.split('従事者')[0]

      # print(o)

      if '，' in o:       # account for multiple occ in one category
        tmp = o.split('，')
        for t in tmp:
          new = get_score(prompt, t, model, tokenizer)
          raw.append((t, new))
          if new > cur_score:     # takes highest probability in group
            cur_score = new
      elif '・' in o:       # account for multiple occ in one category
        tmp = o.split('・')
        for t in tmp:
          new = get_score(prompt, t, model, tokenizer)
          raw.append((t, new))
          if new > cur_score:     # takes highest probability in group
            cur_score = new
      else:
        cur_score = get_score(prompt, o, model, tokenizer)
        raw.append((o, cur_score))

      scores.append(cur_score)  #np.exp(logit))

    return scores, raw

# def get_dist(prompt):
#     scores = [] # keeps max of each row
#     raw = [] # keeps all occupations
#     for o in ['医者', '警察官', '女優', 'ポケモン']:
#       # print(o)

#       # cur_score = get_score(prompt, o)

#       cur_score = 0
#       if '，' in o:       # account for multiple occ in one category
#         tmp = o.split('，')
#         for t in tmp:
#           new = get_score(prompt, t)
#           raw.append((t, new))
#           if new > cur_score:     # takes highest probability in group
#             cure_score = new
#       elif '・' in o:       # account for multiple occ in one category
#         tmp = o.split('・')
#         for t in tmp:
#           new = get_score(prompt, t)
#           raw.append((t, new))
#           if new > cur_score:     # takes highest probability in group
#             cure_score = new
#       else:
#         cur_score = get_score(prompt, o)
#         raw.append((o, cur_score))

#       scores.append(cur_score)  #np.exp(logit))

#     return scores, raw

# prompt = '正一の職業は'
# job_is = []
# job_is_raw = []
# # print(prompt)
# # inputs = tokenizer.encode(prompt, return_tensors='pt') #.cuda()
# scores, raw = get_distribution(prompt)
# job_is.append(scores)
# job_is_raw.append(raw)

# job_is_raw

# job_is

# occupations['job_is_shoichi'] = np.array(job_is[0]).tolist()

# occupations

# scores, raw = get_distribution('正一の仕事は', ['医者', '公安', '泥棒'])

# r = [[raw], [raw], [raw]]
# r

# file = open("test_raw.txt", "w+")
 
# # Saving the array in a text file
# content = str(r)
# file.write(content)
# file.close()

def call_dist(df, mod_name, pmpt_type, occ_csv, mf, model, tokenizer):
  occupations = pd.read_csv(occ_csv)
  occ = occupations['職業']
  job_is_raw = []

  for i in range(len(df)):
  # for i in range(2):
    prompt = df[pmpt_type][i]
    # job_is = []
    
    print(prompt)
    scores, raw = get_distribution(prompt, occ, model, tokenizer)
    # job_is.append(scores)
    occupations[prompt] = np.array(scores).tolist()

    job_is_raw.append(prompt)
    job_is_raw.append(raw)

  occ_name =  mod_name[:4] + '_' + pmpt_type + '_' + mf + '.csv'
  raw_name =  mod_name[:4] + '_' + pmpt_type + '_' + mf + '_raw.txt'
  occupations.to_csv(occ_name, index = False)
  file = open(raw_name, "w+")
  content = (str(job_is_raw))
  file.write(content)
  file.write('\n')
  file.close()

# call_dist(men_pmpt, 'rinna', 'job_is', '/content/drive/MyDrive/constrained/occupations_2020.csv', 'men')



#a = [
#  "rinna/japanese-gpt2-medium" DONE,
#  "gpt2-medium" DONE,
#  "ai-forever/mGPT" DONE,
#  "EleutherAI/gpt-neox-20b",
#  "rinna/japanese-gpt-neox-3.6b", RUNNING    

# rinna neox returned AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'

# pip install protobuf==3.20.1

# tried abeja/gpt-neox-japanese-2.7b but
# abeja
# File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/generation/utils.py", line 1086, in compute_transition_scores
#    indices = sequences[:, cut_idx:] + beam_sequence_indices
#RuntimeError: The size of tensor a (2) must match the size of tensor b (10) at non-singleton dimension 1
#
#


#  "bert-base-multilingual-cased" DONE,
#  "cl-tohoku/bert-base-japanese" DONE]
#


def run_all(mod):

  if 'bert' in mod:
    model = AutoModelForMaskedLM.from_pretrained(mod).cuda()
  else:
    model = AutoModelForCausalLM.from_pretrained(mod).cuda()

  tokenizer = AutoTokenizer.from_pretrained(mod)

  if 'EleutherAI' in mod:
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().cuda()
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")


  call_dist(men_pmpt, mod, 'job_is', 'occupations_2020.csv', 'men', model, tokenizer)
  call_dist(men_pmpt, mod, 'job_type', 'occupations_2020.csv', 'men', model, tokenizer)
  call_dist(men_pmpt, mod, 'job?', 'occupations_2020.csv', 'men', model, tokenizer)

  call_dist(wom_pmpt, mod, 'job_is', 'occupations_2020.csv', 'wom', model, tokenizer)
  call_dist(wom_pmpt, mod, 'job_type', 'occupations_2020.csv', 'wom', model, tokenizer)
  call_dist(wom_pmpt, mod, 'job?', 'occupations_2020.csv', 'wom', model, tokenizer)


# models
a = [
  "rinna/japanese-gpt2-medium",
  "gpt2-medium",
  "ai-forever/mGPT",
  "EleutherAI/gpt-neox-20b",
  "rinna/japanese-gpt-neox-3.6b",
  "bert-base-multilingual-cased",
  "cl-tohoku/bert-base-japanese"]


run_all(a[3]) # neox

run_all(a[4]) # neox-J

# call_dist(men_pmpt, 'job_type', 'occupations_2020.csv', 'men')
# call_dist(men_pmpt, 'job?', 'occupations_2020.csv', 'men')

# call_dist(wom_pmpt, 'job_is', 'occupations_2020.csv', 'wom')
# call_dist(wom_pmpt, 'job_type', 'occupations_2020.csv', 'wom')
# call_dist(wom_pmpt, 'job_?', 'occupations_2020.csv', 'wom')

# run all 6 with other occupations.csv



# file = open("/content/drive/MyDrive/constrained/test_raw.txt", "w+")
 
# # Saving the array in a text file
# content = str(r)
# file.write(content)
# file.close()
