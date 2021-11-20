# A simple analysis tool for Fasttext

This repository contains a simple analysis tool for Fasttext, which replicates the functions about how does fasttext represent a given text, calculate the sentence vector as well as get the final prediction in Python (for official Python wrapper of Fasttext, it finishes these jobs by calling the counterparts in C++ source code). You can get the details about the Besides, it also provides a function that can measure the contribution of each token to the prediction by calculating a contribution score. As a result, by using this function, one can get a rough understanding of the importance of all the words to the full text.    

## Preparation work

According to the source code of official Python wrapper of Fasttext, it does not provide the interface access to the C++ function about how to get the hash value for a given word. So I made a minor modification over the official Fasttext installation by creating a interface accessing to the mentioned C++ function. The followings are the details that what I did:  

1. unzip the source code (.zip file) of the Fasttext (https://github.com/facebookresearch/fastText/releases).

2. insert the following codes to /python/fasttext_module/fasttext/pybind/fasttext_pybind.cc at line 462
    ```c++
    .def(
        "getHash",
        [](fasttext::FastText& m, const std::string word) {
          std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
          return d->hash(word);
        })
    ```
3. insert the following codes to /python/fasttext_module/FastText.py at line 151:
    ```python
    def get_hash(self, word: str):
        """
        Given a word, get its hash value (a uint_32 number).
        """
        return self.f.getHash(word)
    ```

4. Install Python Wrapper of Fasttext by executing:
    ```cmd
    python setup.py install
    ```

## How to use?

The main script is VisualFasttext.py, where a class VisualFasttext is defined. The usage of this class is very simple, first of all, you can create an instance of this class by specifying the path to a supervised fasttext model, and if you want to see the specification of the specified model, you can add a True following the path:

```python
vft_model = VisualFasttext("/path/to/the/supervised_fasttext_model", True) 
```



Then if everything is okey, the specification of the model will be printed, for example:
```
The specifications of the given fasttext model are listed below:
  dim of embedding:  100
  window size:  5
  epoch:  12
  minCount:  100
  wordNgrams:  3
  model:  model_name.supervised
  bucket:  1000000
  minn:  0
  maxn:  3
  learning rate:  0.05
  number of words:  157
  number of labels:  736
```

The above model is trained on the official example dataset(https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz)

Given a text, you can get the sentence_vector by:

```python
sentence_vector = vft_model.get_sentence_vector("input_text")
```



further, if you want to get the predicted labels as well as their scores, you can:

```Python
predicted_labels, predicted_scores = vft_model.predict("input_text")
```

According to my experiments, my code will predict the same labels to the official Fasttext Python wrapper, with having errors around 10^-5 on values.

PS: I've tested my code both on 0.9.1 and 0.9.2, and found it works on both versions.

Please note that currently my code only works on the premise that the fasttext model is trained using softmax as loss function, it is temporarily not compatible with other loss functions. And I'm trying to replicate the hierarchial softmax.

## How to calculate the contribution scores?

The method calculates the contribution scores by collecting the values of all the tokens' hidden states at a certain component which relating to the prediction, and then perform softmax over all the collected values, and finally, the results are regarded as the contribution scores of all the tokens.  

## Reference

[1] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [*FastText.zip: Compressing text classification models*](https://arxiv.org/abs/1612.03651)

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)
