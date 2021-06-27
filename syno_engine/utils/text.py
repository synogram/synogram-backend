import re

re_text = re.compile('[^a-zA-Z0-9?!.,<>\"\'\n\t\s]+')
re_space = re.compile('[\n\t\s]{2,}')

# import enum
# class SP(enum):
#     # BERT tokenizer
#     UNK = '[UNK]'
#     SEP = '[SEP]'
#     PAD = '[PAD]'
#     CLS = '[CLS]'
#     MASK = '[MASK]'

def t2g_clean_text(text):
    # Remove special characters
    cln_text = re.sub(re_text, ' ', text)

    # Split to sentences
    sents = cln_text.split('.')

    def _valid_line(line):
        return all([
            len(line.split(' ')) > 2,
            ])
    
    sents = [re.sub(re_space, ' ', t.strip()) for t in \
            sents if _valid_line(t)]

    return sents
    



if __name__ == '__main__':
    txt = """
        hlel dlkja
        dlkjfkl  *** lkdjfl ka?!@124lkj 
        dkljf sl'j;kljadf;kl
    """
    print(t2g_clean_text(txt))