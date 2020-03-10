import torch
import numpy as np
import pandas as pd

# For initializating this class, first we need the data and the target in the form torch.Tensor and we will also need
# the number of batchs that we want per epoch, we choose this parameter because it makes easier then for coding it

class data_loader_luis():
    def __init__(self,data,targets, n_batch):
        self.num_samples, self.sequence, self.n_vocab = data.shape
        self.data = data 
        self.targets = targets
        self.index = np.arange(self.num_samples)
        self.marker = 0 # this variable is going to take care that we are going through all the samples in each epoch
        self.n_batch = n_batch
        self.batch_size = round(self.num_samples / self.n_batch)
    
    def batch(self):
        # After one round, we are going to try to shuffle the samples so they are not deliver in the same order that 
        # the previous epoch
        
        if self.marker == self.num_samples:
            self.marker = 0
            np.random.shuffle(self.index)
       
        if self.marker + self.batch_size <= self.num_samples:
        
            index_chosen_batch = self.index[self.marker:(self.marker + self.batch_size)]
            self.marker = self.marker + self.batch_size
            data_batch_return = self.data[index_chosen_batch,:,:]
            targets_batch_return = self.targets[index_chosen_batch]
            
            return data_batch_return, targets_batch_return
        
        elif self.marker + self.batch_size > self.num_samples:
            
            leftovers_samples = self.num_samples - self.marker
            
            index_chosen_batch = self.index[self.marker:(self.marker + leftovers_samples)]
            self.marker = self.marker + leftovers_samples
            data_batch_return = self.data[index_chosen_batch,:,:]
            targets_batch_return = self.targets[index_chosen_batch]
            
            return data_batch_return, targets_batch_return

        
        
class vocabulary(object):

    def __init__(self, token_to_idx = None, add_unk = True, unk_token = '<UNK>'):
        
        # Creating the dictionary that will contain the mapping from words to integers
        if token_to_idx is None:
            token_to_index = {}
            
        self._token_to_idx = token_to_index
        
        # We also create a local dictionary for mapping from integers to tokens/words
        self._idx_to_token = {idx : token
                             for token, idx in self._token_to_idx.items()}
        
        # Defining more local variables
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)
        
        
        
    # THIS FUNCTION ALLOWS US TO ADD NEW TOKENS TO OUR DICTIONARY, returns the index of a token in the dictionary if
    # it exists, if it does not, then the token and index are created in both dictionaries
    def add_token(self,token):
            
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
            
        return index
    
    # THIS FUNCTION WILL GIVE US THE INDEX ASSOCIATED TO A GIVEN TOKEN
    # In this funtion, the method get returns the index given the token, if it does not find it then returns whatever it is
    # in the second argument, in this case, the index of the unknown terms
    def lookup_token(self,token):
        if self._add_unk:
            return self._token_to_idx.get(token,self.unk_index)
        else:
            return self._token_to_idx[token]
    
    # THIS FUNCTION WILL GIVE US THE TOKEN ASSOCIATED TO A GIVEN INDEX
    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError('the index (%d) is not in the vocabulary' %index)
        return self._idx_to_token[index]
    
    # IT BRINGS UP SOME SORT OF RESUME OF THE VOCABULARY OBJECT
    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx,
               'add_unk': self._add_unk,
               'unk_token': self._unk_token}
    
    # PENDING TO DEFINE THE REAL FUNTION OF THIS CLASSMETHOD, WAIT FOR THE BIGGER PICTURE TO DEFINE IT

    def __len__(self):
        return len(self._token_to_idx)


    
class dataset_processing():
    
    def __init__(self,train_surname,train_nation,surname_vocab,nation_vocab,number_letters,validation,test,max_sequence):

        self.surname_vocab = surname_vocab
        self.nation_vocab = nation_vocab
        self.number_letters = number_letters
        self.max_sequence = max_sequence
        
        # train
        self.train_surname = train_surname
        self.train_nation = train_nation
        
        # validation
        self.validation_surname = validation.surname
        self.validation_nation = validation.nationality
        
        # test
        self.test_surname = test.surname
        self.test_nation = test.nationality
        
        # train, validation and test datasets encoded
        self.train_surname_encoded = None
        self.validation_surname_encoded = None
        self.test_surname_encoded = None
        
        
        self.train_nation_encoded = None
        self.validation_nation_encoded = None
        self.test_nation_encoded = None

    
    @classmethod
    def create_vocabulary(cls,train,validation,test):
        train_surname = train.surname
        train_nation = train.nationality
        surname_vocab = vocabulary(add_unk = True)
        nation_vocab = vocabulary(add_unk = False)
        max_sequence = 0
        #letter_count = collections.Counter()
        
        for surname in train_surname:
            if len(surname) > max_sequence:
                max_sequence = len(surname)
            for letter in surname:
                surname_vocab.add_token(letter)
                
        
        
        for nation in sorted(set(train_nation)):
            nation_vocab.add_token(nation)
           
        number_letters = len(surname_vocab)
         
        return cls(train_surname,train_nation,surname_vocab,nation_vocab,number_letters,validation,test,max_sequence)
      
    # Transforming the labels into the encoded form
    def encoding_labels(self,dataset_nation):
    
        dataset_nation_encoded = []
        for nation in dataset_nation:
            dataset_nation_encoded.append(int(self.nation_vocab.lookup_token(nation)))
            
        return torch.Tensor(dataset_nation_encoded)

     
        
    def vectorize(self,surname):
        one_hot = np.zeros(self.number_letters)
        for token in surname:
            one_hot[self.surname_vocab.lookup_token(token)] = 1
            #if token not in string.punctuation:
             #   one_hot[self.review_vocab.lookup_token(token)] = 1
        
        return one_hot
        
        
    def transform_data_one_hot_encoded(self,dataset):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        encoded_dataset = torch.zeros(len(dataset),self.max_sequence,self.number_letters).to(device)
        # ,dtype=torch.float64
        for i in range(len(dataset)):
            
            # we are only going to consider those samples whose sequence is below the max sequence established
            if len(dataset[i]) <= self.max_sequence:
                for j in range(len(dataset[i])):
                    encoded_dataset[i][j][self.surname_vocab.lookup_token(dataset[i][j])] = 1

        return encoded_dataset.to(torch.device('cpu'))
        


    def transform_encoded_dataset(self):
        
        self.train_surname_encoded = self.transform_data_one_hot_encoded(self.train_surname)
        self.validation_surname_encoded = self.transform_data_one_hot_encoded(self.validation_surname)
        self.test_surname_encoded = self.transform_data_one_hot_encoded(self.test_surname)

        
        self.train_nation_encoded = self.encoding_labels(self.train_nation)
        self.validation_nation_encoded = self.encoding_labels(self.validation_nation)
        self.test_nation_encoded = self.encoding_labels(self.test_nation)
        
      
    def __len__(self):
        return len(self.surname_vocab)

    
#####################################################
## FUNCTIONS TO STUDY THE PERFORMANCE OF THE MODEL
    
    
def classifier(y_pred):
    y_pred_class = []
    for output in y_pred:
        values, indices = torch.max(output, 0)
        y_pred_class.append(indices)

    return y_pred_class


def accuracy(y_pred,true_target):
    
    y_pred_class = classifier(y_pred)
    true_target = list(true_target)
    count = 0
    for i in range(len(y_pred_class)):
        if int(y_pred_class[i])==int(true_target[i]):
            count += 1
        else:
            pass
    return count/len(y_pred_class)


# y_pred_validation and dataset.validation_nation_encoded
def counfusion_matrix(labels,y_pred,ground_true):
    confusion_matrix = pd.DataFrame(0,index = labels, columns = labels)
    for i in range(len(y_pred)):
        confusion_matrix.iloc[int(ground_true[i]),int(y_pred[i])] += 1
        
    return confusion_matrix