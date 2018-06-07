import string
path = '/home/mayank/Desktop/NeuralNetwork/NLP/Language Model/republic_cleaned.txt'

def load_file(path):
    file = open(path,'r')
    text = file.read()
    file.close()
    return text




def clean_data(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens
    



def get_training_data(path):
    txt = load_file(path)
    tokens = clean_data(txt) 
    length = 51
    sequences = list()
    for i in range(length, len(tokens)):   
        seq = tokens[i-length:i]
        # convert into a line
        line = ' '.join(seq)
        sequences.append(line)
    return sequences
    
    

sequences = get_training_data(path)

# save this list to a file
# here "republic_sequences.txt"