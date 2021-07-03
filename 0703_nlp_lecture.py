def pad_punctuation(sentence, punc):
      for p in punc:
    sentence = sentence.replace(p, ' '+p+' ')

  return sentence

sentence = 'Hi, my name is john.'

print(pad_punctuation(sentence, ['.', '?', '!', ',']))