import pytest
import ce2
    
def test_get_texts_and_labels_shape():
    out = ce2.get_texts_and_labels()
    assert len(out) == 2
    
def test_get_texts_and_labels_type():
    out = ce2.get_texts_and_labels()
    assert isinstance(out, tuple)
    
def test_texts_type():
    out = ce2.get_texts_and_labels()
    texts = out[0]
    assert all(isinstance(element, str) for element in texts)

def test_labels_type():
    out = ce2.get_texts_and_labels()
    labels = out[1]
    assert all(isinstance(element, str) for element in labels)

def test_read_files_shape():
    text_lst = []
    fnames = []
    path = 'News Articles/The Wrap'
    out = ce2.read_files(path, text_lst, fnames)
    assert len(out) == 2

def test_read_files_type():
    text_lst = []
    fnames = []
    path = 'News Articles/The Wrap'
    out = ce2.read_files(path, text_lst, fnames)
    assert isinstance(out, tuple)
    
def test_text_lst_type():
    text_lst = []
    fnames = []
    path = 'News Articles/The Wrap'
    out = ce2.read_files(path, text_lst, fnames)
    text_lst = out[0]
    assert all(isinstance(element, str) for element in text_lst)
    
def test_fnames_type():
    text_lst = []
    fnames = []
    path = 'News Articles/The Wrap'
    out = ce2.read_files(path, text_lst, fnames)
    fnames = out[1]
    assert all(isinstance(element, str) for element in fnames)
    
