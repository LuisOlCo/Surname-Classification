# Surname-Classification

The purpose of this project is the classification of surnames into corrsponding nationalities. This project is approached using three different techniques. First of all, since we are going to create one-hot-encoded vectors, we do not want to consider A and a different element but the contrary, this is why the data will be processed, turning all the upper cases into lower cases. Then the data will be split proportionally into three datasets.

### Techniques
The first technique, is the simplest approach, the surname will be split into their letters, a one-hot-encoded vector will be created, this vector will have as many elements as different letters are found in the training set, the one-hot-encoded vector will have 1 in those elements that the represent a letter that is contained in the surname. As we can observe, this approach is ignoring the sequence of this elements and the frequency of the letters in the surname. 

The second technique is slightly different to the first one, the procedure is exactly the same, but in this approach we give a vector that has the same number of elements as it had in the first one, but instead of placing 1 onthe element that is present in the surname, we indicate the number of times that that element is in the surname
