

```python
import nlp
```

## Load Models
For Word2VecModelUser, we are loading simple toy model, thus it will limits a lot of vocabularies and won't capture actual meaning of the words. Also SummaryModelUser will download ~ 1.5 GB model on first load if not already cached.


```python
nlp.Word2VecModelUser.load()
nlp.SummaryModelUser.load()
```

## Sematic Field using Word2VecModelUser
Finding most similar words, and provide score and vector representation of the words.


```python
nlp.Word2VecModelUser.sematic_field('standard', 'and', dim=2, topn=10)
```




    [('And', 1000, [0.05930540710687637, -0.007837523706257343]),
     ('Standard', 1000, [-0.05686343088746071, 0.008029039949178696]),
     ('.', 0.045971684054911865, [0.027088096365332603, -0.010154887102544308]),
     ('A', 0.03816488569502843, [0.042034801095724106, 0.024351894855499268]),
     ('Are', 0.05600392847433968, [-0.0035735880956053734, -0.005412881728261709]),
     ('At', 0.06342861535862275, [-0.03349757939577103, -0.0038572095800191164]),
     ('By', 0.052770899220119696, [-0.002200002083554864, -0.012464993633329868]),
     ('Can', 0.05006940034577702, [-0.009932102635502815, -0.002771513070911169]),
     ('For', 0.056281460179011325, [-0.006776257883757353, 0.014417946338653564]),
     ('Have', 0.04993749924308995, [-0.015585360117256641, -0.004299872554838657])]



## Summarizing Article using SummaryModelUser
SummaryModelUser uses https://pypi.org/project/bert-extractive-summarizer/ which utilizes HuggingFace Pytorch transformers library to run extractive summarizations. SummaryModelUser for now is simple wrapper for these libraries.


```python
article = """
A computer is a machine that can be instructed to carry out sequences of arithmetic or logical 
operations automatically via computer programming. Modern computers have the ability to follow 
generalized sets of operations, called programs. These programs enable computers to perform
an extremely wide range of tasks. A "complete" computer including the hardware, the 
operating system (main software), and peripheral equipment required and
used for "full" operation can be referred to as a computer system. This term may as well be
used for a group of computers that are connected and work together, in particular a computer network or
computer cluster.

Computers are used as control systems for a wide variety of industrial and consumer devices.
This includes simple special purpose devices like microwave ovens and remote controls, factory devices
such as industrial robots and computer-aided design, and also general purpose devices like personal
computers and mobile devices such as smartphones. The Internet is run on computers and it connects hundreds
of millions of other computers and their users.
"""
```


```python
summary = nlp.SummaryModelUser.summarize(article)
summary.replace('\n', ' ')
```




    'A computer is a machine that can be instructed to carry out sequences of arithmetic or logical  operations automatically via computer programming. These programs enable computers to perform an extremely wide range of tasks.'




```python

```
