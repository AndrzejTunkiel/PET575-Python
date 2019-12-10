# PET575, Data Analysis in Python

*(Open in Playground mode. Shift+Enter runs a cell.)*

## What is Python?

* Python is a programmng language
* It is completely free
* It is the most popular language in Machine Learning and Data Analysis
* One of the most popular languages in general

## How is Python different?

##### No need to declare a type of variable:


```python
a = 5
b = 3.4
c = "test"

print("Variable 'a' is: " + str(type(a)))
print("Variable 'b' is: " + str(type(b)))
print("Variable 'c' is: " + str(type(c)))
```

    Variable 'a' is: <class 'int'>
    Variable 'b' is: <class 'float'>
    Variable 'c' is: <class 'str'>
    

##### Function definitions and loops do not use brackets, only whitespace. This ensures that code looks tidy.


```python
def print_n_powers_of_2(n): #function definition
    for i in range(n):      #simple loop
        print(2**i)         #note that python starts counting at zero
        
print_n_powers_of_2(6)      #execute the function passing number 6 as parameter n
```

    1
    2
    4
    8
    16
    32
    

##### Python can be run in different ways:
* *notebook* - just like here
    * Required Anaconda installation (https://www.anaconda.com) - recommended
    * Online on Google Colaboratory (https://colab.research.google.com)
* .py file
    * Python3 is already installed on MacOS
    * Python3 is already installed on Linux
    * Python3 can installed on Windows through Anaconda (https://www.anaconda.com)
    
    
Note: Use Python 3.x only. Python 2.x also exists, but is being depreciated. For most intents and purposes Python 2 code will work in Python 3.

##### Working with notebooks

* memory state is not saved between sessions, but the output is
* cells can be executed out of sequence


```python
a = a + 1  #add one to a defined earlier. Re-running this cell will keep incrementing a
print (a)
```

    6
    

##### Why Python over Matlab?

* Python has much higher adoption in the industry
* Python pushes out MATLAB in academia too
* Python is **free**, MATLAB is very much not. Many companies simply do not, and will not, own a MATLAB license.

Note: Python is very distributed in nature compared to compact package of MATLAB. If you are searching for funciton - use your search engine of choice:  Google/DuckDuckGo/Qwant

## Python basics

### Defining variables

Below are some basic examples of variable definition


```python
a = 5        #python automatically knows if it is int, float, str, etc
b = "rabbit" #strings (text) comes in quote marks
c = a + 2    #you can do operations at the same time as definition
```

### Printing output

*Printing* is the basic way of showing data in Python. It is very forgiving.


```python
print(a)
print(b)
print(c+a)
```

    5
    rabbit
    12
    

### Lists, NumPy

Lists, arrays, matrices - a sequence of variables


```python
a = [0, 1, 2, 3]
print(a)
```

    [0, 1, 2, 3]
    


```python
a = [1, "two", 3.0] #feel free to mix types
print(type(a))
print()
print(type(a[0]))
print(type(a[1]))
print(type(a[2]))
```

    <class 'list'>
    
    <class 'int'>
    <class 'str'>
    <class 'float'>
    

To work on such data structures, the best is to use a library called **numpy**.


```python
import numpy as np
```


```python
a = [0, 1, 2, 3]
a = np.asarray(a) #convert python list to a numpy array
print(a)          #note the different printing format than before
```

    [0 1 2 3]
    

This allows us to do different operations easily:


```python
print(a+2)
```

    [2 3 4 5]
    


```python
print(a*2)
```

    [0 2 4 6]
    


```python
b = np.asarray([5,6,5,6])
print(a)
print("    +")
print(b)
print("    =")

print(a+b)
```

    [0 1 2 3]
        +
    [5 6 5 6]
        =
    [5 7 7 9]
    

#### Slices

To get a part of an array, slices are used.


```python
print(f'This is a complete array,                    a:{a}')
print(f'This is a a single element in an array,   a[2]:{a[2]}')
print(f'This is a slice of an array,            a[0:2]:{a[0:2]}') #note that slice is [inclisive:exclusive]
print(f'This is a last element of an array,      a[-1]:{a[-1]}')
```

    This is a complete array,                    a:[0 1 2 3]
    This is a a single element in an array,   a[2]:2
    This is a slice of an array,            a[0:2]:[0 1]
    This is a last element of an array,      a[-1]:3
    

Arrays can be nested


```python
#defining a nested list
a = [
    [1,2,3,4],
    [5,6,7,8]
    ]
```

**Caution**: Python lists behave differently than numpy arrays!


```python
print(a[0][1])
```

    2
    

Notice different notation when converted to numpy array


```python
a = np.asarray(a)
print(a[0,1])
```

    2
    

Slices are particularly useful here


```python
print(a[1,1:3])
```

    [6 7]
    

### CAUTION!
Lists and arrays behave differently than variables.


```python
a = 2
b = a
b = b + 2

print(f'a is equal to {a}')
print(f'b is equal to {b}')
```

    a is equal to 2
    b is equal to 4
    

**But**, if we do the same thing with lists:


```python
a = [1,1,1,1,1]
b = a
b[1] = 3

print(f'List a looks like this: {a}')
print(f'List b looks like this: {b}')
```

    List a looks like this: [1, 3, 1, 1, 1]
    List b looks like this: [1, 3, 1, 1, 1]
    

Notice that list **a** also changed. Let's see how numpy arrays behave:


```python
a = np.asarray([1,1,1,1,1])
b = a
b[1] = 3

print(f'Numpy array a looks like this: {a}')
print(f'Numpy array b looks like this: {b}')
```

    Numpy array a looks like this: [1 3 1 1 1]
    Numpy array b looks like this: [1 3 1 1 1]
    

Same thing. There may be differences though:


```python
a = [1,1,1,1,1]
b = a[1:3]
b[1] = 3

print(f'List a looks like this: {a}')
print(f'List b looks like this: {b}')

a = np.asarray([1,1,1,1,1])
b = a[1:3]
b[1] = 3

print(f'Numpy array a looks like this: {a}')
print(f'Numpy array b looks like this: {b}')
```

    List a looks like this: [1, 1, 1, 1, 1]
    List b looks like this: [1, 3]
    Numpy array a looks like this: [1 1 3 1 1]
    Numpy array b looks like this: [1 3]
    

When using a *slice* **list** created a copy, but **numpy array** kept the reference.

Numpy can make a copy of an array if you are explicit about it


```python
a = np.asarray([1,1,1,1,1])
b = np.copy(a)
b[1] = 3

print(f'Numpy array a looks like this: {a}')
print(f'Numpy array b looks like this: {b}')
```

    Numpy array a looks like this: [1 1 1 1 1]
    Numpy array b looks like this: [1 3 1 1 1]
    

### NumPy continued
##### Let's generate some random numbers

* Average: 10
* Standard deviation: 2
* Shape of output: 30

**NOTE: you are NOT to memorize functions, use Google/DuckDuckGo/Qwant!**

https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html


```python
a = np.random.normal(10, 2, 100) #center, std, size
print(a)
```

    [ 7.6240235  11.76280582  9.01762748 10.42444908  8.37928248 11.9039274
      9.63897752  9.30555676 13.91692845  8.70449799 12.65109028  7.11773653
     10.76471188  6.97133693 10.56894209  9.63599316  7.45851747 10.12269343
     11.83791389  8.18845118 10.46332721  9.33313022 10.30906221  9.36794416
      9.11778192 10.05807366 10.93251319 11.02177274  9.87430969  9.10077282
     11.97647627 10.87875783 12.90546025  6.75260154 11.73262604  7.40087435
      8.86020838  7.66627408 10.65263508 11.7352162  12.15964735  9.09293901
     10.21105915  9.99854962  9.62507197 12.35789567  6.41393447  9.15983168
      7.58104186  9.5203138   8.36059366 13.50784819 10.67476131  8.9913917
      8.62791446  9.93447487  9.43521536  6.78106859 13.76038286  8.1894918
      8.84594017 12.62527636 11.04222497 13.17259629  9.34019203  8.43976763
      8.96223989 10.71134532 11.13620578  8.53631264 14.36170845  8.9480509
     12.2032383   7.85834879 11.11436957 11.50781356 11.65684675 10.15723462
     11.39501347  9.48313993 10.1582522   9.67430299  9.61059347 14.54208797
      5.57595916  8.82337766  7.66922658 10.1213115   4.98167652  6.70006533
     10.00018063 13.00862328 12.63482425 11.05492175 13.54941216  9.53367974
      8.81358687 10.18805671  7.24579629 10.5156783 ]
    

Numpy allows us to quickly calculate many attributes of a given array


```python
a_max = a.max()
print(f'The maximum value is {a_max}') #note the f notation
print(f'The minimum value is {a.min()}') #we can do more in the curly brackets too
print(f'The sum is {a.sum()}')
print(f'The average is {a.mean()}')
print(f'The standard deviation is {a.std()}')
```

    The maximum value is 14.542087965625658
    The minimum value is 4.981676524063535
    The sum is 996.4181873140388
    The average is 9.964181873140388
    The standard deviation is 1.9644008543827691
    

To avoid too many decimals, we can simply round the numbers


```python
np.round(a.std(),2)
```




    1.96



### Functions

Functions work the same way as in most other programming languages


```python
def a_to_b(a,b):
    return(a**b)

print(a_to_b(3,4))
```

    81
    

### Conditional statements

same as functions. Remember dual equal signs!


```python
def even_odd(a):
    if a%2 == 0:    #Percentage symbol is modulo. Here divide by 2 and return the rest
        print(f'{a} is even')
    else:
        print(f'{a} is odd')
        
even_odd(3)
even_odd(4)
```

    3 is odd
    4 is even
    

### Loops

#### for

for loop can be used in a number of ways


```python
for i in range(4):
    print(i)
```

    0
    1
    2
    3
    


```python
for i in range(4,10):
    print(i)
```

    4
    5
    6
    7
    8
    9
    


```python
text = "rabbit"
for i in text:
    print(i)
```

    r
    a
    b
    b
    i
    t
    


```python
some_numbers = np.random.normal(1,1,10)
for i in some_numbers:
    print(np.round(i,2))
```

    -0.3
    1.29
    -0.21
    -0.62
    0.26
    2.04
    1.48
    0.89
    0.16
    1.46
    

## Excercise 1

You are playing a simple board game. You roll a dice and advance forward by the number of dots. Easy.

Some fields are special. Some are *You wait one round* or *You roll again*.

One field is *very* special. The rules state:
* roll dice 3 times
* you move forward by the two lowest numbers
* you move back by the highest number

Example: you roll 2, 6 and 4. You move forwards by 2 + 4 and move backwards by 6

**Question:** Is it a good, or a bad field? What is the mean and median result?


```python
def monte_carlo(rolls):
    
    results = []   #initiate an empty list
    
    for i in range(rolls):  #Perform given number of simulations
        sample = np.random.randint(1,7,3) #Return random integers from low (inclusive) to high (exclusive).
        sample = np.sort(sample) #sort the values ascending
        results.append(sample[0]+sample[1]-sample[2])  #apply the rule and append it to the list
        
    return(results)  #return the results
```

Execute the simulation. You can see how increasing the amount of iterations increases the calculation time


```python
a = monte_carlo(10000)
```

Print requested values


```python
print(f'The mean movement was {np.mean(a)}')
print(f'The median movement was {np.median(a)}')
print(f'Achieved values are {np.unique(a)}')
```

    The mean movement was 0.5754
    The median movement was 1.0
    Achieved values are [-4 -3 -2 -1  0  1  2  3  4  5  6]
    

### Data visualization

For making of various charts **MatPlotLib** is the most popular library. It has a similar syntax to MATLAB.

**Seaborn** is an add-on to **MatPlotLib** that makes generating more complex charts easier.

First, let's import the libraries:


```python
import matplotlib.pyplot as plt #matplotlib is the most common plotting library for Python.
import seaborn as sns   #seaborn is an extension to matplotlib


sns.set() #this function call sets some defaults to seaborn's defaults.
```

Let's plot a distribution plot


```python
bins = len(np.unique(a))    #define variable "bins" by counting how many unique values there were in the simulation
sns.distplot(a, bins = bins, hist=True, kde=False, norm_hist=True) #make a histogram
plt.xticks(np.unique(a))    #defining which ticks should be plotted on x axis
plt.show()     #call to show the plot. Note that we mixed seaborn and matplotlib functions. 
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_67_0.png)


Experiment on your own: Does it make a difference how many sides does the dice have?

### More visualization

Let's generate some data to plot


```python
n = 10000
theta = np.random.normal(np.pi*3/4, 1, n)  #random numbers with normal distribution, centered at 3/4 pi
noise_x = np.random.normal(0,0.2,n) #generating numpy arrays with noise
noise_y = np.random.normal(0,0.2,n)
x = np.sin(theta)+noise_x #calculating x coordinate
y = np.cos(theta)+noise_y #calculating y coordinate
```

We can plot the data as a **joint plot** with **hex plot** in the middle. This is very convinient for showing density data.

Seaborn makes creating such relatively complex charts very easy.


```python
sns.jointplot(x, y, kind="hex", gridsize=50, cmap="inferno")
```




    <seaborn.axisgrid.JointGrid at 0x249aa537cf8>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_72_1.png)


Alternatively we can use **kernel density estimate** for a smooth plot.


```python
sns.jointplot(x=x, y=y, kind="kde",n_levels=35, cmap="viridis", cbar=True)
```




    <seaborn.axisgrid.JointGrid at 0x249aa789048>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_74_1.png)


Some plots can be combined. Note slightly different notation for generating the plot.


```python
n = 1000
theta = np.random.normal(np.pi*3/4, 0.6, n)
noise_x = np.random.normal(0,0.1,n)
noise_y = np.random.normal(0,0.1,n)
x = np.sin(theta)+noise_x
y = np.cos(theta)+noise_y

noise_v = np.random.normal(0,0.2,n)
noise_w = np.random.normal(0,0.2,n)
v = np.sin(theta)*0.2+noise_v
w = np.cos(theta)*0.2+noise_w


ax = sns.kdeplot(x, y, shade=True, cmap="Blues", shade_lowest=False)
ax = sns.kdeplot(v, w, shade=True, cmap="Reds", shade_lowest=False)
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_76_0.png)


### Line plot, scatter plot


```python
n = 100
x = np.linspace(0,10,n) #create an array that starts at 0, ends at 10 and has n evenly spaced elements
y = np.sin(x)
```

Very simple plotting with just MatPlotLib


```python
plt.plot(y)
```




    [<matplotlib.lines.Line2D at 0x249aaa17400>]




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_80_1.png)


Seaborn requires two parameters. Notice x axis ticks. In the example above they correspond to the index of the numpy array.


```python
sns.lineplot(x,y)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249aaa4e278>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_82_1.png)


Best practice is to specify both axis, see example why:


```python
x = np.random.uniform(0,10,n)   #create an array with random, uniform distribution
                                #between 0 and 10
x = np.sort(x)                  #sort. This is now an unevenly polled domain

y = np.sin(x)

plt.plot(y)
```




    [<matplotlib.lines.Line2D at 0x249a9514f98>]




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_84_1.png)



```python
plt.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x249a9507748>]




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_85_1.png)


The first chart is distorted due to uneven polling, even though the signal is perfectly noise free. When specified x and y variables, the signal is now plotted correctly

### Scatter plots

Very similar to line plots, work better with noisy data


```python
n = 200
x = np.linspace(0,10,n) #create an array that starts at 0, ends at 10 and has n evenly spaced elements
y = np.sin(x)+np.random.normal(0,0.1,n)

outlier_x = np.random.uniform(0,10,30)
outlier_y = np.random.uniform(-1.5, 1.5, 30)

x = np.append(x, outlier_x)
y = np.append(y, outlier_y)

sns.lineplot(x,y)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249aab4d438>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_87_1.png)


Line plot looks very bad, compare with scatter plot


```python
sns.scatterplot(x,y)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249aabad198>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_89_1.png)


One can plot multiple plots at the same time


```python
n = 500
x = np.linspace(1,100,n)
y1 = np.cumsum(np.random.normal(0,1,n))
y2 = np.cumsum(np.random.normal(-0.1,1,n))
y3 = np.cumsum(np.random.normal(0,2,n))

sns.lineplot(x,y1, label="First one")
sns.lineplot(x,y2, label="Second one")
sns.lineplot(x,y3, label="Third one")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249aabea898>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_91_1.png)


### Other plot types

There are many more plot types, see documentation for MatPlotLib and Seaborn for more information


```python
n = 500
x = np.linspace(1,40,n)
y = 0.3*x+np.sin(x*10)

plt.polar(x,y)
```




    [<matplotlib.lines.Line2D at 0x249aac6e978>]




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_93_1.png)


# Pandas

Pandas is an extremely popular package in data science. In short, it deals with 2D arrays called **dataframes** and has a plethora of data science functions.

Let's start by exploring the Titanic dataset


```python
import numpy as np  #import numpy
import pandas as pd  #import pandas
import seaborn as sns #import seaborn
import matplotlib.pyplot as plt #import matplotlib
df = sns.load_dataset('titanic') #load the titanic dataset to a pandas dataframe
sns.set() #seaborn's defaults
```

It is very simple to look around the dataframe. Use .head() to view 5 top rows. You can also use .head(*a*) for defined number of rows.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



.describe() returns basic statistical data for each column


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



To get a list of all the attributes in a dataset call list(df).


```python
list(df)
```




    ['survived',
     'pclass',
     'sex',
     'age',
     'sibsp',
     'parch',
     'fare',
     'embarked',
     'class',
     'who',
     'adult_male',
     'deck',
     'embark_town',
     'alive',
     'alone']



**Sorting.**

It is sometimes convinient to sort the DataFrame by an attribute. Note that in example below the original dataframe remains untouched (unsorted). 


```python
df.sort_values(by="age").head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>803</th>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>0.42</td>
      <td>0</td>
      <td>1</td>
      <td>8.5167</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>755</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>0.67</td>
      <td>1</td>
      <td>1</td>
      <td>14.5000</td>
      <td>S</td>
      <td>Second</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>644</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>19.2583</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>469</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>19.2583</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>0.83</td>
      <td>0</td>
      <td>2</td>
      <td>29.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>831</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>0.83</td>
      <td>1</td>
      <td>1</td>
      <td>18.7500</td>
      <td>S</td>
      <td>Second</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>305</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>0.92</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>S</td>
      <td>First</td>
      <td>child</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>827</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>1.00</td>
      <td>0</td>
      <td>2</td>
      <td>37.0042</td>
      <td>C</td>
      <td>Second</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>381</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>1.00</td>
      <td>0</td>
      <td>2</td>
      <td>15.7417</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>164</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1.00</td>
      <td>4</td>
      <td>1</td>
      <td>39.6875</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>183</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>1.00</td>
      <td>2</td>
      <td>1</td>
      <td>39.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>child</td>
      <td>False</td>
      <td>F</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>386</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1.00</td>
      <td>5</td>
      <td>2</td>
      <td>46.9000</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>172</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>1.00</td>
      <td>1</td>
      <td>1</td>
      <td>11.1333</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>788</th>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1.00</td>
      <td>1</td>
      <td>2</td>
      <td>20.5750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>642</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>2.00</td>
      <td>3</td>
      <td>2</td>
      <td>27.9000</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



To check who is the oldest on Titanic, sort descending. Note additional .dropna() that drops the lines containing missing values.


```python
df.dropna().sort_values(by="age", ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>34.6542</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>745</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>71.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>B</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>1</td>
      <td>61.9792</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>B</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>456</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.5500</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>E</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>438</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>263.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>C</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>275</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>63.0</td>
      <td>1</td>
      <td>0</td>
      <td>77.9583</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>D</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>252</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.5500</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>C</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>625</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>32.3208</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>D</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>170</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>33.5000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>B</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>587</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>60.0</td>
      <td>1</td>
      <td>1</td>
      <td>79.2000</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>B</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>366</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>60.0</td>
      <td>1</td>
      <td>0</td>
      <td>75.2500</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>D</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.5500</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>487</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>29.7000</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>B</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>268</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>1</td>
      <td>153.4625</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



To drop NaN existing only in the age column, more complex operation is necessary.

Notice that we got more results - the deck attribute is full of NaNs


```python
df2 = df[np.isfinite(df['age'])]                        #create a new dataframe, that is a copy of df where df['age']
                                                        #is finite, ie. is not NaN or inf
                                                        #Note! This is a reference, not a copy!
df2.sort_values(by="age", ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>851</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7750</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>49.5042</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>34.6542</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>116</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>70.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>672</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>70.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>745</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>71.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>B</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>66.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>1</td>
      <td>61.9792</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>B</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>280</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>456</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.5500</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>E</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>438</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>263.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>C</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>545</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>64.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>275</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>63.0</td>
      <td>1</td>
      <td>0</td>
      <td>77.9583</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>D</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>483</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>63.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.5875</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



We can quickly analyse the age distribution among the passangers


```python
df = df[np.isfinite(df['age'])]                                                 #get only results with finite age
sns.distplot(df[df['who'] == "man"]['age'], bins=10, hist=True, label="men")    #histogram of men
sns.distplot(df[df['who'] == "woman"]['age'], bins=10, hist=True, label="women")#histogram of women
plt.legend()                                                                    #add a legend to chart
```




    <matplotlib.legend.Legend at 0x249aace6550>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_109_1.png)


We can also quickly show how that distribution is when split into different classes.


```python
df1 = df[df['class']=="First"]      #defining sub-dataframes containing only passengers of first/second/third class
df2 = df[df['class']=="Second"]
df3 = df[df['class']=="Third"]

sns.distplot(df1[df1['who'] == "man"]['age'], hist=True, label="men, 1st class")
sns.distplot(df1[df1['who'] == "woman"]['age'], hist=True, label="women, 1st class")
plt.legend()
plt.show()               #necessary to show the chart. If we ommit this, we will have everything in one chart.

sns.distplot(df2[df2['who'] == "man"]['age'], hist=True, label="men, 2nd class")
sns.distplot(df2[df2['who'] == "woman"]['age'], hist=True, label="women, 2nd class")
plt.legend()
plt.show()

sns.distplot(df3[df3['who'] == "man"]['age'], hist=True, label="men, 3rd class")
sns.distplot(df3[df3['who'] == "woman"]['age'], hist=True, label="women, 3rd class")
plt.legend()
plt.show()
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_111_0.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_111_1.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_111_2.png)


Things to notice: 
- The bins are not equal
- The bins do not overlap
- Kernel Density Estimate can be misleading


```python
sns.distplot(df1[df1['who'] == "woman"]['age'], bins= 10, hist=True, kde=False, norm_hist=True, label="Histogram")

sns.distplot(df1[df1['who'] == "woman"]['age'], hist=False, label="KDE, bw: 2",
             kde_kws={'gridsize' : 100, "bw" : 2})

sns.distplot(df1[df1['who'] == "woman"]['age'], hist=False, label="KDE, bw: 5",
             kde_kws={'gridsize' : 100, "bw" : 5})

sns.distplot(df1[df1['who'] == "woman"]['age'], hist=False, label="KDE, bw: 15",
             kde_kws={'gridsize' : 100, "bw" : 15})

plt.legend()
```




    <matplotlib.legend.Legend at 0x249abdb41d0>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_113_1.png)


### Why not simply Excel?

Prep takes longer, but reuse is **much** faster with Python.

Let's load weather data from Lerwick weather station (Shetland Islands, Scotland).


```python
df = pd.read_csv("http://www.ux.uis.no/~atunkiel/lerwickdata.csv", delim_whitespace=True)
                                                                #delim_whitespace - whitespace separated file
                #notice that we read csv straight from a website

#year and month are separate in this dataset. Let's merge it and tell pandas that it is a date.
df["date"] = df["mm"].map(str) + "." +df["yyyy"].map(str)
df['date'] = pd.to_datetime(df['date'],format = '%m.%Y')

```

Quick inspection of the dataframe


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yyyy</th>
      <th>mm</th>
      <th>tmax[degC]</th>
      <th>tmin[degC]</th>
      <th>af[days]</th>
      <th>rain[mm]</th>
      <th>sun[hours]</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1930</td>
      <td>12</td>
      <td>7.0</td>
      <td>3.6</td>
      <td>0</td>
      <td>122.4</td>
      <td>13.1</td>
      <td>1930-12-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1931</td>
      <td>1</td>
      <td>4.9</td>
      <td>0.2</td>
      <td>13</td>
      <td>108.0</td>
      <td>29.7</td>
      <td>1931-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1931</td>
      <td>2</td>
      <td>4.4</td>
      <td>-0.3</td>
      <td>12</td>
      <td>138.3</td>
      <td>52.6</td>
      <td>1931-02-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1931</td>
      <td>3</td>
      <td>4.2</td>
      <td>-1.0</td>
      <td>17</td>
      <td>18.2</td>
      <td>135.7</td>
      <td>1931-03-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1931</td>
      <td>4</td>
      <td>8.1</td>
      <td>2.5</td>
      <td>1</td>
      <td>70.9</td>
      <td>134.7</td>
      <td>1931-04-01</td>
    </tr>
  </tbody>
</table>
</div>



Let's inspect the maximum recorded temperatures


```python
sns.scatterplot(df['date'],df['tmax[degC]'])
```

    C:\Users\2921228\AppData\Local\Continuum\anaconda3\lib\site-packages\pandas\plotting\_converter.py:129: FutureWarning: Using an implicitly registered datetime converter for a matplotlib plotting method. The converter was registered by pandas on import. Future versions of pandas will require you to explicitly register matplotlib converters.
    
    To register the converters:
    	>>> from pandas.plotting import register_matplotlib_converters
    	>>> register_matplotlib_converters()
      warnings.warn(msg, FutureWarning)
    




    <matplotlib.axes._subplots.AxesSubplot at 0x249abd5aac8>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_119_2.png)


Let's do some rolling average to have a cleaner look at data.

How wide should be the window? Let's try 10, 20, 30, ... , 100 days


```python
for i in range(1,10):
    sns.scatterplot(df['date'],df['tmax[degC]'].rolling(10*i).mean(center=True), 
                    alpha=0.3, label=f'rolling average {10*i}')
    plt.show()
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_0.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_1.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_2.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_3.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_4.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_5.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_6.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_7.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_121_8.png)


It seems that window of 60 gives the best results. Why 60? Probably because there are 12 months in a year and 60 divides nicely by 12.

Let's check the theory by plotting rolling window with widths of 12, 24, 36, ... , 120.


```python
for i in range(1,10):
    sns.scatterplot(df['date'],df['tmax[degC]'].rolling(12*i).mean(center=True),
                    alpha=0.3, label=f'rolling average {12*i}')
    plt.show()
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_0.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_1.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_2.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_3.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_4.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_5.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_6.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_7.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_123_8.png)


This explains some articles from the 70s:
- U.S. Scientist Sees New Ice Age Coming (The Washington Post, July 9, 1971)
- Ice Age Around the Corner (Chicago Tribune, July 10, 1971)
- New Ice Age Coming – It’s Already Getting Colder (L.A. Times, October 24, 1971)

Pollution was likely driving factor in the cooling. Currently we somewhat *overcompensated* with CO2 emissions.

We can explore some correlations, like how is temperature related to rain?


```python
x= df['tmax[degC]']
y= df['rain[mm]']
sns.jointplot(x, y, kind="kde", levels=25, cmap="inferno")
```




    <seaborn.axisgrid.JointGrid at 0x249ac3efe10>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_126_1.png)


We can fairly easily plot LOWESS:
- **Lo**cally
- **We**ighted
- **S**catterplot
- **S**moothing


```python
from statsmodels.nonparametric.smoothers_lowess import lowess #import library that will do all the heavy lifting
y = df['tmax[degC]']
x = np.arange(0,len(df['date']),1) #converting into linear space from 0 to length of the dataset
w = lowess(y, x, frac=1/5)         #calculating LOWESS
                                   #frac is the window width in terms of a fraction of full width.
```

And let's plot the results. Some calculations on the *x* values to recover the year.


```python
sns.scatterplot(x/12+1930,y.rolling(12).mean(center=True),linewidth=0,color="r",alpha=0.3)
sns.lineplot(w[:,0]/12+1930,w[:,1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249ac357f28>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_130_1.png)


We can explore how much smooting is too much smoothing


```python
plt.figure(figsize=(15,10)) # bigger plot for visibility

sns.scatterplot(x,y.rolling(12).mean(center=True),linewidth=0,color="r",alpha=0.3) #scatterplot for reference


for i in reversed(range(1,21)):  #note that reversed is used. This is to change the order of plotting of the charts.
    w = lowess(y, x, frac=1/(i*2))
    
    if i != 20:
        sns.lineplot(w[:,0],w[:,1], hue=i, palette='viridis',hue_norm=(1,20), legend=False)
    else:
        sns.lineplot(w[:,0],w[:,1], hue=i, palette='viridis',hue_norm=(1,20))

plt.plot()
```




    []




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_132_1.png)


Violin plots are often better than box plots, as they carry more data


```python
x = df['rain[mm]']
y = df['mm']
sns.violinplot(y,x)
plt.show()
sns.boxplot(y,x)
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_134_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x249ac1685f8>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_134_2.png)


### Real drilling data

Let's consider some real drilling data. We can import from web again, straight out of a zip file.


```python
df=pd.read_csv("http://www.ux.uis.no/~atunkiel/Norway-NA-15_$47$_9-F-9%20A%20depth.zip")

```


```python
list(df)
```




    ['Unnamed: 0',
     'Unnamed: 0.1',
     'Measured Depth m',
     'TOFB s',
     'AVG_CONF unitless',
     'MIN_CONF unitless',
     'Average Rotary Speed rpm',
     'STUCK_RT unitless',
     'Corrected Surface Weight on Bit kkgf',
     'Corrected Total Hookload kkgf',
     'MWD Turbine RPM rpm',
     'MWD Raw Gamma Ray 1/s',
     'MWD Continuous Inclination dega',
     'Pump 2 Stroke Rate 1/min',
     'Rate of Penetration m/h',
     'Bit Drill Time h',
     'Corrected Hookload kkgf',
     'MWD Gravity Toolface dega',
     'Mud Density Out g/cm3',
     'MWD GR Bit Confidence Flag %',
     'Pump Time h',
     'PowerUP Shock Rate 1/s',
     'Total SPM 1/min',
     'HKLO kkgf',
     'Average Hookload kkgf',
     'Total Hookload kkgf',
     'DRET unitless',
     'Extrapolated Hole TVD m',
     'MWD Gamma Ray (API BH corrected) gAPI',
     'Pump 1 Strokes unitless',
     'ROPIH s/m',
     'EDRT unitless',
     'Pump 1 Stroke Rate 1/min',
     'Rate of penetration m/h',
     'Iso-butane (IC4) ppm',
     'Bit Drilling Run m',
     'MWD Total Shocks unitless',
     'Total Bit Revolutions unitless',
     'Nor-butane (NC4) ppm',
     'Lag Depth (TVD) m',
     'Pump 2 Strokes unitless',
     'Corr. Drilling Exponent unitless',
     'Total Vertical Depth m',
     'Ethane (C2) ppm',
     'TMP In degC',
     'Mud Density In g/cm3',
     'n-Penthane ppm',
     'Mud Density In g/cm3.1',
     'Weight on Bit kkgf',
     'Bit Revolutions  (cum) unitless',
     'Averaged WOB kkgf',
     'Hole Depth (TVD) m',
     'MWD Shock Risk unitless',
     'Bit run number unitless',
     'RHX_RT unitless',
     'Pump 3 Strokes unitless',
     'Total Strokes unitless',
     'Inverse ROP s/m',
     'Pass Name unitless',
     'Pump 4 Stroke Rate 1/min',
     'Iso-pentane (IC5) ppm',
     'Rig Mode unitless',
     'MWD Shock Peak m/s2',
     'Flow Pumps L/min',
     'SPN Sp_RigMode 2hz unitless',
     'Average Standpipe Pressure kPa',
     'Bit Depth m',
     'Rate of Penetration (5ft avg) m/h',
     'Gas (avg) %',
     'Propane (C3) ppm',
     'String weight (rot,avg) kkgf',
     'TOBO s',
     'AJAM_MWD unitless',
     'Tank volume (active) m3',
     '1/2ft ROP m/h',
     'Weight On Hook kkgf',
     'Hole depth (MD) m',
     'Mud Flow In L/min',
     'BHFG unitless',
     'Temperature Out degC',
     'Averaged TRQ kN.m',
     'MWD Continuous Azimuth dega',
     'RGX_RT unitless',
     'MWD DNI Temperature degC',
     'Bit Drilling Time h',
     'HKLI kkgf',
     'Average Surface Torque kN.m',
     'Methane (C1) ppm',
     'MWD Magnetic Toolface dega',
     'Total Downhole RPM rpm',
     'SHK3TM_RT min',
     'Elapsed time in-slips s',
     'Stand Pipe Pressure kPa',
     'Pump 3 Stroke Rate 1/min',
     'Averaged RPM rpm',
     'Pump 4 Strokes unitless',
     'Inverse ROP (5ft avg) s/m',
     'S1AC kPa',
     'S2AC kPa',
     'IMWT g/cm3',
     'OSTM s',
     'nameWellbore',
     'name',
     'IMP/ARC Attenuation Conductivity 40-in. at 2 MHz mS/m',
     'ARC Annular Pressure kPa',
     'MWD Collar RPM rpm',
     'IMP/ARC Non-BHcorr Phase-Shift Resistivity 28-in. at 2 MHz ohm.m',
     'IMP/ARC Phase-Shift Conductivity 40-in. at 2 MHz mS/m',
     'Annular Temperature degC',
     'IMP/ARC Non-BHcorr Phase-Shift Resistivity 40-in. at 2 MHz ohm.m',
     'ARC Gamma Ray (BH corrected) gAPI',
     'IMP/ARC Non-BHcorr Attenuation Resistivity 40-in. at 2 MHz ohm.m',
     'MWD Stick-Slip PKtoPK RPM rpm',
     'IMP/ARC Non-BHcorr Attenuation Resistivity 28-in. at 2 MHz ohm.m',
     'IMP/ARC Phase-Shift Conductivity 28-in. at 2 MHz mS/m']



Verify the contents of the DataFrame. Note that real-life datasets can be overrun with NaNs.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>Measured Depth m</th>
      <th>TOFB s</th>
      <th>AVG_CONF unitless</th>
      <th>MIN_CONF unitless</th>
      <th>Average Rotary Speed rpm</th>
      <th>STUCK_RT unitless</th>
      <th>Corrected Surface Weight on Bit kkgf</th>
      <th>Corrected Total Hookload kkgf</th>
      <th>...</th>
      <th>MWD Collar RPM rpm</th>
      <th>IMP/ARC Non-BHcorr Phase-Shift Resistivity 28-in. at 2 MHz ohm.m</th>
      <th>IMP/ARC Phase-Shift Conductivity 40-in. at 2 MHz mS/m</th>
      <th>Annular Temperature degC</th>
      <th>IMP/ARC Non-BHcorr Phase-Shift Resistivity 40-in. at 2 MHz ohm.m</th>
      <th>ARC Gamma Ray (BH corrected) gAPI</th>
      <th>IMP/ARC Non-BHcorr Attenuation Resistivity 40-in. at 2 MHz ohm.m</th>
      <th>MWD Stick-Slip PKtoPK RPM rpm</th>
      <th>IMP/ARC Non-BHcorr Attenuation Resistivity 28-in. at 2 MHz ohm.m</th>
      <th>IMP/ARC Phase-Shift Conductivity 28-in. at 2 MHz mS/m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>273.101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>273.253</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>273.406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>273.558</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>273.710</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 115 columns</p>
</div>



We can explore the data through charts, tweaking them to show data best


```python
df = df[np.isfinite(df['Measured Depth m'])]
df = df[np.isfinite(df['MWD Continuous Inclination dega'])]


plt.figure(figsize=(20,10))
sns.scatterplot(df['Measured Depth m'], df['MWD Continuous Inclination dega'],
                linewidth=0,
                hue=df['Average Hookload kkgf'],
                hue_norm=(85,100),
                palette="viridis",
                size=df['Average Surface Torque kN.m'],
                size_norm=(0,10),
                sizes=(20,300)
                )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249ac18acc0>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_141_1.png)


Utilize the .describe() function to gauge the normalization range


```python
df['Average Hookload kkgf'].describe()

```




    count    1963.000000
    mean       94.636537
    std         3.698203
    min        71.966965
    25%        91.820703
    50%        96.034577
    75%        97.359066
    max       104.303565
    Name: Average Hookload kkgf, dtype: float64



Simple plot is also useful to understand the data contents, such as outliers.


```python
df['Average Hookload kkgf'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249ac38df28>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_145_1.png)


## Color palettes

While in publication grayscale is typically preferred, when using color use good color schemes. Examples below are good - the brightness of colors is consistent with color scale.


```python
palettes = ['viridis','inferno', 'gray', 'Blues', 'Reds', 'hot', 'cividis']
for i in palettes:
    sns.palplot(sns.color_palette(i,10))
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_147_0.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_147_1.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_147_2.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_147_3.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_147_4.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_147_5.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_147_6.png)


Divergin palletes are also possible to highlight that something is below or above a certain level (like a physical map)


```python
sns.palplot(sns.diverging_palette(220, 20, n=11))
sns.palplot(sns.diverging_palette(145, 280, s=85, l=25, n=11))
sns.palplot(sns.diverging_palette(255, 133, l=60, n=11, center="dark"))
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_149_0.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_149_1.png)



![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_149_2.png)


#### Avoid "jet" palette

A ~rainbow pallete that is the most commonly used despite being a poor in representing data

<img src="http://jakevdp.github.io/figures/jet.png">

https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/ 

https://pdfs.semanticscholar.org/ee79/2edccb2c88e927c81285344d2d88babfb86f.pdf


Seaborn will straight out refuse to use it returning **ValueError: *No.***


```python
#palettes = ['jet']
#for i in palettes:
#    sns.palplot(sns.color_palette(i))
```

Jet colormap will be even more confusing when converted to grayscale.

Don't worry about the code below, it is there just to generate the chart.


```python
def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)



%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 6,300)
y = np.linspace(0, 3,300)[:, np.newaxis]
z = 10 * np.cos(x ** 2) * np.exp(-y)
cmaps = [plt.cm.jet, grayify_cmap('jet'), plt.cm.gray]
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.subplots_adjust(wspace=0)

for cmap, ax in zip(cmaps, axes):
    im = ax.imshow(z, cmap=cmap)
    ax.set_title(cmap.name)
    fig.colorbar(im, ax=ax)
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_153_0.png)


Use a palette that has lightness corresponding to value for better results


```python
cmaps = [plt.cm.CMRmap, grayify_cmap('CMRmap'), plt.cm.gray]
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.subplots_adjust(wspace=0)

for cmap, ax in zip(cmaps, axes):
    im = ax.imshow(z, cmap=cmap)
    ax.set_title(cmap.name)
    fig.colorbar(im, ax=ax)
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_155_0.png)


## Regression

In Python we can easily perform regression using any function.

Let's assume we have a signal governed by an equation
\begin{equation*}
y = -0.2x^3 + 3x^2+30sin(x) 
\end{equation*}

This signal is noisy. Noise has a standard deviation of 30.

We have total 100 samples to work with, but 20 of them are outliers


```python
np.random.seed(10) #fixed random seed, so you always get the same set of random numbers

n = 20 #outlier count
x_0 = np.linspace(-10,10,100-n)  #generate 100 minus n evenly spaced values from -10 to 10
y_0 = -0.2*x_0**3+3*x_0**2+30*np.sin(x_0) #calculate y

sns.lineplot(x_0,y_0,color="red", label="ground truth") #plot clean function

x = x_0  #keep x_0 separate for plotting later
y = y_0 + np.random.normal(0,30,100-n)  #generate new function - same as clean one plus noise with sd=30
sns.scatterplot(x,y,label="truth+noise") #plot the noisy chart

outliers_x = np.random.uniform(-12,12,n) #generate n outliers, x coordinates
outliers_y = np.random.uniform(-50,500,n)#... y coordinates
sns.scatterplot(outliers_x,outliers_y,label="outliers") #plot outliers only

x = np.concatenate([x, outliers_x]) #add the outliers to the x array
y = np.concatenate([y, outliers_y]) #add the outliers to the y array
```


![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_157_0.png)


Let's look at the raw data. You can see, that the data is very noisy and difficult to make a regression of!


```python
sns.scatterplot(x,y,label="outliers",linewidth=0)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249acd90d30>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_159_1.png)


Let's assume that we know the function itself, but we do not know the parameters, effectively, known shape of the funtion is:
\begin{equation*}
y = a_0 x^3 + a_1 x^2+a_2sin(x) 
\end{equation*}

We define a function in a parametric way that is suitable for the algorithm that will come later. 


```python
def fun(a, x, y):
    return a[0]*x**3+a[1]*x**2+a[2]*np.sin(x) - y
```

We can import an optimization function from SciPy based on least squares method


```python
from scipy.optimize import least_squares
```

Let's develop two models. 

One is optimized for outliers, where they no longer carry such a heavy weight (typical issue with least squares method).

Second is a vanilla Least Squares method


```python
a0 = np.ones(3) #initiating the parameters with ones

model = least_squares(fun, a0, args=(x,y),loss='soft_l1') #the soft_l1 is the more lenient loss function

model2 = least_squares(fun, a0, args=(x,y))

print(model.x) #show the calculated parameters
```

    [-0.20864698  2.95738833 31.35128843]
    

Model returned:

$a_0 = -0.20864698$ (originally $-0.2$)

$a_1 =  2.95738833$ (originally $3$)

$a_2 = 31.35128843$ (originally $30$)

Which is very close to original values. Let's inspect it graphically. The match is nearly perfect.





```python
plt.figure(figsize=(10,7)) #let's use slightly bigger chart
sns.scatterplot(x,y, label="input data",s=100)

sns.lineplot(x_0,y_0,color="red", lw=2,alpha=1,label="ground truth")

y2 = model.x[0]*x**3+model.x[1]*x**2+model.x[2]*np.sin(x) #calculate y based on the model
sns.lineplot(x, y2, color="green",lw=5,alpha=0.5, label="model, optimized")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249accfd9e8>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_167_1.png)


In comparison, vanilla Least Square method did not perform so well:


```python
plt.figure(figsize=(10,7))
sns.scatterplot(x,y, label="input data",s=100)

sns.lineplot(x_0,y_0,color="red", lw=2,alpha=1,label="ground truth")

y3 = model2.x[0]*x**3+model2.x[1]*x**2+model2.x[2]*np.sin(x)
sns.lineplot(x, y3, color="green",lw=2,dashes="-",alpha=1, label="model, vanilla")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249aa9a2048>




![png](PET575%2C_Data_Analysis_in_Python_files/PET575%2C_Data_Analysis_in_Python_169_1.png)


Let's calculate $R^2$ values for both models. 

Note: in this implementation $R^2$ value can be negative


```python
from sklearn.metrics import r2_score

print(f'R^2 value for optimized fitting is: {r2_score(y,y2)}' )
print(f'R^2 value for standard fitting is : {r2_score(y,y3)}' )


```

    R^2 value for optimized fitting is: 0.13286411606455029
    R^2 value for standard fitting is : 0.2922538200391831
    

The objectively much better model has a lower $R^2$ value. As you can see, this is not the ultimate metric. 

k means clustering

feature engineering

Correlation

Classification


```python

```


```python

```


```python

```
