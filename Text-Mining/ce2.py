import os

def get_texts_and_labels():
    """read txt files from pos/neg directories 
       and attach pos/neg as labels

    Returns
    -------
    X : A list of str 
        text file contents, input data for training
    y : A list of str
        the corresponding sentiment of X, labels
    
    """
    # Initialize train data and train labels
    X = [] 
    y = []

    # Read text data from directory
    path = "review_polarity/pos"
    for fname in os.listdir(path):
            f = open(os.path.join(path, fname), 'r') 
            content = f.read()        
            X.append(content)
            y.append("pos")

    path = "review_polarity/neg"
    for fname in os.listdir(path):
            f = open(os.path.join(path, fname), 'r') 
            content = f.read()
            X.append(content)
            y.append("neg")
    
    return X,y


def read_files(path, text_lst, fnames):
    """read txt files from a directory
       add contents and file names to lists

    Parameters
    ----------
    path : String
        the directory path of txt file(s)
    text_lst : A list of str 
        a list of text file contents
    fnames : A list of str 
        a list of the corresponding file names 

    Returns
    -------
    text_lst : expanded text_lst
    fnames : expanded fnames
    """  
    
    for fname in os.listdir(path):
        if (fname =='.DS_Store'or fname =='.ipynb_checkpoints'):
             continue
        content = open(os.path.join(path, fname), 'rb').read().decode('utf-8', errors='ignore')      
        text_lst.append(content)
        fnames.append(fname)
        
    return text_lst,fnames
