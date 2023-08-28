#!/usr/bin/env python
# coding: utf-8

# Q1. Student grades and absence
# Download the dataset from Kaggle:
# 
# https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
# 
# Compute the probability that a student obtains 80% or more in mathematics, given he/she misses atleast 10 classes. Do this in Python.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('student-mat.csv')
df.head(3)


# In[ ]:


len(df)


# Add a boolean column called grade_A noting if a student achieved 80% or higher as a final score. Original values are on a 0–20 scale so we multiply by 5.

# In[ ]:


df


# In[ ]:


x = df['G3']*5
x


# In[ ]:


pivot = df.iloc[0:395][['absences','G3']]
pivot


# In[ ]:


count = len(pivot[(pivot['absences']>10) &(pivot['G3']>16)])
print(count)


# In[ ]:


count1 = len(pivot[pivot['absences']>10])
print(count1)


# In[ ]:


count2 = len(pivot[(pivot['G3']>16)])
print(count2)


# 

# In[ ]:


P_a_intersection_b = count/395
P_a = count2/395
P_b = count1/395


# In[ ]:


P = P_a_intersection_b/P_b
P


# ## Second Way to implement

# In[ ]:


df['grade_A'] = np.where(df['G3']*5 >= 80, 1, 0)


# Make another boolean column called high_absenses with a value of 1 if a student missed 10 or more classes.

# In[ ]:


df['high_absenses'] = np.where(df['absences'] >= 10, 1, 0)


# Add one more column to make building a pivot table easier.

# In[ ]:


df['count'] = 1


# In[ ]:


df


# In[ ]:


df = df[['grade_A','high_absenses','count']]
df.head()


# In[ ]:


pd.pivot_table(
    df,
    values='count',
    index=['grade_A'],
    columns=['high_absenses'],
    aggfunc=np.size,
    fill_value=0
)


# In our case:
# P(A) is the probability of a grade of 80% or greater.
# P(B) is the probability of missing 10 or more classes.
# P(A|B) is the probability of a 80%+ grade, given missing 10 or more classes.

# Calculations of parts:
# P(A) = (35 + 5) / (35 + 5 + 277 + 78) = 0.10126582278481013
# P(B) = (78 + 5) / (35 + 5 + 277 + 78) = 0.21012658227848102
# P(A ∩ B) = 5 / (35 + 5 + 277 + 78) = 0.012658227848101266

# P(A|B) = 0.012658227848101266/ 0.21012658227848102= 0.06
# 
# There we have it. The probability of getting at least an 80% final grade, given missing 10 or more classes is 6%.

# Ques 2. **THE BIRTHDAY PROBLEM**
# Compute the probability of getting a minimum of one overlapping birthday in a random group of 23 peoples.
# Because ontaining random sample again and again is tedious task, we can do simulations on a computer with assumptions.
# The birthdays are independent of each other. Each possible birthday has the same probability. There are only 365 possible birthdays (not 366, as ignoring the leap year.
# Hint : in other words, we're modelling the process as drawing 23 independent samples from a discrete uniform distribution with parameter n = 365.

# In[ ]:





# In[ ]:


from random import randint


NUM_PEOPLE = 23
NUM_POSSIBLE_BIRTHDAYS = 365
NUM_TRIALS = 10000


def generate_random_birthday():
    birthday = randint(1, NUM_POSSIBLE_BIRTHDAYS)
    return birthday


def generate_k_birthdays(k):
    birthdays = [generate_random_birthday() for _ in range(k)]
    return birthdays


def aloc(birthdays):
    unique_birthdays = set(birthdays)

    num_birthdays = len(birthdays)
    num_unique_birthdays = len(unique_birthdays)
    has_coincidence = (num_birthdays != num_unique_birthdays)

    return has_coincidence


def estimate_p_aloc():
    num_aloc = 0
    for _ in range(NUM_TRIALS):
        birthdays = generate_k_birthdays(NUM_PEOPLE)
        has_coincidence = aloc(birthdays)
        if has_coincidence:
            num_aloc += 1

    p_aloc = num_aloc / NUM_TRIALS
    return p_aloc


p_aloc = estimate_p_aloc()
print(f"Estimated P(ALOC) after {NUM_TRIALS} trials: {p_aloc}")


# **Ques 3. The weight of certain species of frog is uniformly distributed from 15 and 25 grams. if you randomly select a frog, what is the probability that the frog weights between 17 and 19 grams.**

# In[1]:


from scipy.stats import uniform
uniform.cdf(x=19, loc = 15, scale =10) -uniform.cdf(x=17, loc =15, scale=10)

p=(19-17)/(25-15)
print(p)


# **Ques 4. Telecommunication Industry**
# According to the Telecommunication Industry the average monthly cell phone bill is Rs. 1000 with a standard deviation of Rs. 200.
# 
# What is the probability that a randomly selected cell phone bill is more than Rs 1200? What is the probability that a randomly selected cell phone bill is between Rs 750 and Rs 1200? What is the probability that a randomly selected cell phone bill is no more than Rs 650? What is the amount above which lies top 15% of cell phone bills? What is the amount below which lies bottom 25% of cell phone bills?
# 
# Note: This is a problem of normal probability distribution. Though the distribution is not mentioned, in absence of any other information we assume normality in the population.

# In[ ]:


from scipy import stats
a = stats.norm.cdf(1200,1000,200)
g = 1-a
print(g)

b= stats.norm.cdf(750,1000,200)
print(a-g)
stats.norm.ppf(.15,1000,200)

stats.norm.ppf(.25,1000,200)


# In[ ]:





# Q5. Fruit problem
# Suppose we own a fruit shop and on an average 3 customers arrive in the shop every 10 minutes. The mean rate here is 3 or λ = 3. Poisson probability distributions can help us answer questions like what is the probability that 5 customers will arrive in the next 10 mins?

# In[ ]:


from scipy.stats import poisson
poisson.cdf(k=5, mu=3)


# 
