# Say "Hello, World!" With Python
print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 != 0:
        print("Weird")
    elif (n % 2 == 0 and (n >= 2 and n <= 5)):
        print("Not Weird")
    elif (n % 2 == 0 and (n >= 6 and n <= 20)):
        print("Weird")
    else:
        print("Not Weird")
        
        

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a // b)
    print(a / b)

# Loops
if __name__ == '__main__':
    n = int(input())
    i = 0
    for i in range(n):
        print(i * i)

# Write a function
def is_leap(year):
    leap = False
    
    if year % 4 == 0:
        leap = True
        if year % 100 == 0:
            leap = False
            if year % 400 == 0:
                leap = True
    return leap

# Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range(1, n+1):
        print(i, end = '')

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    result = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n]
    print(result)
    
    

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = max([x for x in arr if x != max(arr)])
    print(result)

# Nested Lists
if __name__ == '__main__':
    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])
    second = sorted(set([score for name, score in students]))[1]
    result = sorted([name for name, score in students if score == second])
    for n in result:
        print(n)
    
    
    
    
        

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    marks = list(student_marks[query_name])
    result = sum(marks)/len(marks)
    print(f"{result:.2f}")

# Lists
if __name__ == '__main__':
    N = int(input())
    my_list = []
    for i in range(N):
        command = input().split()
        if command[0] == "append":
            my_list.append(int(command[1]))
        elif command[0] == "insert":
            my_list.insert(int(command[1]), int(command[2]))
        elif command[0] == "print":
            print(my_list)
        elif command[0] == "sort":
            my_list.sort()
        elif command[0] == "remove":
            my_list.remove(int(command[1]))
        elif command[0] == "pop":
            my_list.pop()
        else:
            my_list.reverse()
        
             
            
    
            
    
    

# Tuples
if __name__ == '__main__':
    n = int(input())
    l = map(int, input().split())
      
      
    my_tuple = tuple(l)
    print(hash(my_tuple))
    

# sWAP cASE
def swap_case(s):
    result = ""
    for c in s:
        if c >= 'a' and c <= 'z':
            result += c.upper()
        elif c >= 'A' and c <= 'Z':
            result += c.lower()
        else: 
            result += c
    return result
    

# String Split and Join
def split_and_join(line):
    # write your code here
    result = ""
    for c in line:
        if c == ' ':
            result += '-'
        else:
            result += c
    return result

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    # Write your code here
    print(f"Hello {first} {last}! You just delved into python.")

# Mutations
def mutate_string(string, position, character):
    result = string[:position] + character + string[position + 1:]
    return result
    

# Find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)):
        if string[i:].startswith(sub_string):
            count += 1
    return count
    

# String Validators
if __name__ == '__main__':
    s = input()
    print(any(z.isalnum() for z in s))
    print(any(z.isalpha() for z in s))
    print(any(z.isdigit() for z in s))
    print(any(z.islower() for z in s))
    print(any(z.isupper() for z in s))

# Text Alignment
#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = map(int, input().split())
pattern = '.|.'
for i in range(n // 2):
    j = 2 * i + 1
    print((pattern * j).center(m, '-'))
print('WELCOME'.center(m, '-'))
k = n // 2 - 1
while k >= 0:
    j = 2 * k + 1
    print((pattern * j).center(m, '-'))
    k -= 1
    

# String Formatting
def print_formatted(number):
    width = len(bin(n)[2:])
    for i in range(1, number + 1):
        decim = str(i)
        octal = oct(i)[2:]
        hexad = hex(i)[2:]
        binar = bin(i)[2:]
        
        print(decim.rjust(width), octal.rjust(width), hexad.upper().rjust(width), binar.rjust(width))  
    

# Alphabet Rangoli
def print_rangoli(size):
    l = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    m = len('-'.join(l[size-1::-1] + l[1:size]))
    for i in range(1,size):
        print('-'.join(l[size-1:size-i:-1] + l[size-i:size]).center(m, '-'))
    for i in range(size, 0, -1):
        print('-'.join(l[size-1:size-i:-1] + l[size-i:size]).center(m,'-'))


# Capitalize!

# Complete the solve function below.
def solve(s):
    for i in s.split():
        s = s.replace(i, i.capitalize())
    return s


# The Minion Game
def minion_game(string):
    vow = 0
    cons = 0
    for i in range(len(string)):
        if string[i] not in 'AEIOU':
            cons += len(string) - i
        else:
            vow += len(string) - i
    if vow > cons:
        print('Kevin', vow)
    elif vow < cons:
        print('Stuart', cons)
    else:
        print('Draw')  


# Merge the Tools!
def merge_the_tools(string, k):
    for i in range(len(string) // k):
        print(''.join(dict.fromkeys(string[i * k: (i * k) + k])))

# Introduction to Sets
def average(array):
    my_s = set(array)
    avg = sum(my_s) / len(my_s)
    
    return (avg)

# No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOU
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == "__main__":
    n, m = input().split()
    arr = input().split()
    A = set(input().split())
    B = set(input().split())
    
    print(sum([(i in A) - (i in B) for i in arr]))
        

# Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
n1 = int(input())
set_a = set(map(int, input().split()))
n2 = int(input())
set_b = set(map(int, input().split()))
a = (set_a.difference(set_b))
b = (set_b.difference(set_a))
result = a.union(b)
for i in sorted(result):
    print(i)

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
countries = set()
for i in range(N):
    countries.add(input())
print(len(countries))

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N=int(input())
for i in range(N):
    command = input().split()
    if command[0] == "remove":
        s.remove(int(command[1]))
    elif command[0] == "discard":
        s.discard(int(command[1]))
    else:
        s.pop()
print(sum(s))

# Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = input()
arr_n = set(map(int, input().split()))
b = input()
arr_b = set(map(int, input().split()))
inter = arr_n.intersection(arr_b)
print(len(arr_n) + len(arr_b) - len(inter))

# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.intersection(b)))

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n1 = int(input())
first = set(map(int, input().split()))
n2 = int(input())
second = set(map(int, input().split()))
print(len(first-second))

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.symmetric_difference(b)))

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
A = set(map(int, input().split()))
n_other_sets = int(input())
for _ in range(n_other_sets):
    oper = input().split()[0]
    new_set = set(map(int, input().split()))
    
    if oper == "intersection_update":
        A.intersection_update(new_set)
    elif oper == "update":
        A.update(new_set)
    elif oper == "symmetric_difference_update":
        A.symmetric_difference_update(new_set)
    elif oper == "difference_update":
        A.difference_update(new_set)
        
print(sum(A))

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
k = int(input())
myarr = list(map(int, input().split()))
myset = set(myarr)
print(((sum(myset) * k) - (sum(myarr))) // (k - 1))

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
for i in range(int(input())):
    _, a = input(), set(input().split())
    _, b = input(), set(input().split())
    print(b.intersection(a) == a)

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
a = set(map(int, input().split()))
res = "True"
for i in range(int(input())):
    b = set(map(int, input().split()))
    if len(b) > len(b.intersection(a)):
        res = "False"
        
print(res)

# collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter as ctr
x = input()
shoe_sizes = ctr(map(int, input().split()))
res = 0
for _ in range(int(input())):
    size, price = tuple(map(int, input().split()))
    if shoe_sizes.get(size):
        res = res + price
        shoe_sizes[size] = shoe_sizes[size] - 1
print(res)

# DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
d = defaultdict(list)
n, m = map(int, input().split())
for i in range(n):
    d[input()].append(str(i + 1))
    
for _ in range(m):
    t = input()
    if d[t]:
        print(" ".join(d[t]))
    else:
        print(-1)

# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
total = 0
n = int(input())
cols = input().split()
for _ in range(n):
    students = namedtuple('student', cols)
    MARKS, CLASS, NAME, ID = input().split()
    student = students(MARKS, CLASS, NAME, ID)
    total += int(student.MARKS)
    
print(total / n)

# Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
ord_dict = OrderedDict()
n = int(input())
for i in range(n):
    item, price = input().rsplit(' ', 1)
    ord_dict[item] = ord_dict.get(item, 0) + int(price)
for item in ord_dict:
    print(item, ord_dict[item])

# Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
count = {}
words = []
for _ in range(n):
    word = input()
    words.append(word)
    
    if word in count:
        count[word] += 1
    else:
        count[word] = 1
    
print(len(count))
for word in count:
    print(count[word], end = ' ')

# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
d = deque()
n = int(input())
for i in range(n):
    com = input().split()
    if com[0] == 'append':
        d.append(int(com[1]))
    elif com[0] == 'appendleft':
        d.appendleft(int(com[1]))
    elif com[0] == 'pop':
        d.pop()
    elif com[0] == 'popleft':
        d.popleft()
        
for v in d:
    print(str(v), end = ' ')

# Company Logo
from collections import Counter
if __name__ == '__main__':
    s = input()
    occur = Counter(list(sorted(s)))
    
    for ch, occ in occur.most_common(3):
        print(ch, occ)

# Arrays

def arrays(arr):
    arr1 = arr[::-1]
    res = numpy.array(arr1, float)
    return res

# Shape and Reshape
import numpy
arr = list(map(int, input().split()))
my_array = numpy.array(arr)
print(my_array.reshape(3, 3))

# Transpose and Flatten
import numpy
n, m = map(int, input().split())
arr = list()
for _ in range(n):
    arr.append(input().split())
array = numpy.array(arr, int)
print(array.transpose())
print(array.flatten())

# Concatenate
import numpy
n, m, p = map(int, input().split())
l1 = list()
l2 = list()
for _ in range(n):
    l1.append(input().split())
    
for _ in range(m):
    l2.append(input().split())
    
arr1 = numpy.array(l1, int)
arr2 = numpy.array(l2, int)
print(numpy.concatenate((arr1, arr2)))


# Zeros and Ones
import numpy as np
my_tuple = tuple(map(int, input().split()))
print(np.zeros(my_tuple, dtype=int))
print(np.ones(my_tuple, dtype=int))

# Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
print(numpy.eye(n, m))
# print numpy.eye(*map(int,raw_input().split()))

# Array Mathematics
import numpy as np
n, m = map(int, input().split())
A = np.array([list(map(int, input().split())) for _ in range(n)])
B = np.array([list(map(int, input().split())) for _ in range(n)])
print(np.add(A, B), np.subtract(A, B), np.multiply(A, B), np.floor_divide(A, B), np.mod(A, B), np.power(A, B), sep='\n')

# Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
arr = numpy.array(input().split(), float)
print(numpy.floor(arr), numpy.ceil(arr), numpy.rint(arr), sep="\n")


# Sum and Prod
import numpy
n, m = map(int, input().split())
l = list()
for _ in range(n):
    l.append(input().split())
A = numpy.array(l, int)
print(numpy.prod(numpy.sum(A, axis=0), axis=0))

# Min and Max
import numpy
arr = []
n, m = map(int, input().split())
for _ in range(n):
    row = list(map(int, input().split()))
    arr.append(row)
    
matr = numpy.array(arr)
res = numpy.min(matr, axis=1)
print(max(res))

# Mean, Var, and Std
import numpy as np
n, m = map(int, input().split())
matr = np.array([input().split() for _ in range(n)], int)
print(np.mean(matr, axis=1))
print(np.var(matr, axis=0))
print(round(np.std(matr, axis=None), 11))

# Dot and Cross
import numpy
n = int(input())
a = numpy.array([input().split() for _ in range(n)], int)
b = numpy.array([input().split() for _ in range(n)], int)
print(numpy.dot(a, b))

# Inner and Outer
import numpy as np
A = np.array(input().split(), int)
B = np.array(input().split(), int)
print(np.inner(A, B))
print(np.outer(A, B))

# Polynomials
import numpy
n = list(map(float, input().split()))
m = int(input())
print(numpy.polyval(n, m))

# Linear Algebra
import numpy
numpy.set_printoptions(legacy='1.13')
n = int(input())
a = numpy.array([input().split() for _ in range(n)], float)
print(numpy.linalg.det(a))

# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for _ in range(int(input())):
    a = input()
    res = bool(re.match(r'^[+-]?[0-9]*\.[0-9]+$', a))
    print(res)

# Re.split()
regex_pattern = r"[.,]"	# Do not delete 'r'.

# Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT 
import re
m = re.search(r'([a-zA-Z0-9])\1', input())
if m:
    print(m.group(1))
else:
    print(-1)

# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
pattern = r"(?<=[^aeiou])([aeiou|AEIOU]{2,})(?=[^aeiou])"
res = re.findall(pattern, input())
for a in res:
    print(a)
    
if not res:
    print(-1)

# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
k = input()
pattern = re.compile(k)
res = pattern.search(S)
if not res:
    print("(-1, -1)")
    
while res:
    print("({0}, {1})".format(res.start(), res.end() - 1))
    res = pattern.search(S, res.start() + 1)

# Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
for _ in range(n):
    res = input()
    while ' || ' in res or ' && ' in res:
        res = res.replace(" || ", " or ")
        res = res.replace(" && ", " and ")
    
    print(res)

# Validating Roman Numerals
thousand = 'M{0,3}'
hundred = '(C[MD]|D?C{0,3})'
ten = '(X[CL]|L?X{0,3})'
digit = '(I[VX]|V?I{0,3})'
regex_pattern = r"%s%s%s%s$" % (thousand, hundred, ten, digit) 
# Do not delete 'r'.

# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for _ in range(int(input())):
    if re.match(r'[789]\d{9}$', input()):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import email.utils
import re
 
for _ in range(int(input())):
    l = input()
    parse = email.utils.parseaddr(l)[1]
    
    res = re.match(r"(^[A-Za-z][A-Za-z0-9\._-]+)@([A-Za-z]+)\.([A-Za-z]{1,3})$", parse)
    
    if res:
        print(l)

# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for _ in range(int(input())):
    l = input()
    res = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', l)
    
    for i in res:
        if i != '':
            print(i)

# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        if attrs:
            for n, m in attrs:
                print("->", n, ">", m)
                
    def handle_endtag(self, tag):
        print("End   :", tag)
        
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        if attrs:
            for n, m in attrs:
                print("->", n, ">", m)
parser = MyHTMLParser()
for i in range(int(input())):
    parser.feed(input())

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
            
        print(data)
        
    def handle_data(self, data):
        if data != "\n":
            print(">>> Data")
            
            print(data)
  
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def print_atributes(self, attrs):
        for name, val in attrs:
            print(f'-> {name} > {val}')
            
    def handle_starttag(self, tag, attrs):
        print(tag)
        self.print_atributes(attrs)
parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

# Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
pattern = r'^(?!.*(.).*\1)(?=(?:.*[A-Z]){2,})(?=(?:.*\d){3,})[a-zA-Z0-9]{10}$'
lst = []
res = int(input())
for _ in range(res):
    l = input()
    result = re.search(pattern, l)
    lst.append(result)
for n in lst:
    if n:
        print("Valid")
    else:
        print("Invalid")

# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
l = re.compile(r'(?!.*(\d)(-?\1){3})(?=^[456])(([0-9]{4}-?){4})$')
for _ in range(int(input().strip())):
    if l.search(input().strip()):
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes
regex_integer_in_range = r"^[1-9][0-9]{5}$"	
# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(?=(\d).\1)"
# Do not delete 'r'.

# Matrix Script
#!/bin/python3
import math
import os
import random
import re
import sys
n, m = map(int, input().split())
l = list()
b = ""
for _ in range(n):
    l.append(input())
for c in zip(*l):
    b += "".join(c)
print(re.sub(r'(?<=[A-Za-z0-9])[^A-Za-z0-9]+(?=[A-Za-z0-9])', ' ', b))

# Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
for _ in range(int(input())):
    _, queue = input(), deque(map(int, input().split()))
    res = True
    
    for cube in reversed(sorted(queue)):
        if queue[0] == cube:
            queue.popleft()
        elif queue[-1] == cube:
            queue.pop()
        else:
            res = False
            break
    
    if res:
        print("Yes")
    else:
        print(("No"))

# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a // b)
    
    except ValueError as e:
        print("Error Code:", e)
        
    except ZeroDivisionError as e:
        print("Error Code:", e)
    

# Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
m, d, y = map(int, input().split())
day = calendar.weekday(y, m, d)
print(calendar.day_name[day].upper())

# Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime
# Complete the time_delta function below.
def time_delta(t1, t2):
    t1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    res = str(int(abs((t1-t2).total_seconds())))
    
    return res
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, x = map(int, input().split())
res = []
for _ in range(x):
    res.append(map(float, input().split()))
for st in zip(*res):
    print(sum(st)/len(st))

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
S = input()
upper = sorted([ch for ch in S if ch.isupper()])
lower = sorted([ch for ch in S if ch.islower()])
odd = sorted([d for d in S if d.isdigit() if int(d) % 2 != 0])
even = sorted([d for d in S if d.isdigit() if int(d) % 2 == 0])
print("".join(lower + upper + odd + even))

# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    arr.sort(key = lambda arr: arr[k])
    
    for d in arr:
        print(*d)

# Map and Lambda Function
cube = lambda x: x**3
# complete the lambda function
def fibonacci(n):
    arr = []
    for i in range(n):
        if i == 0 or i == 1:
            arr.append(i)
        else:
            arr.append(arr[i-1] + arr[i - 2])
    return arr

# XML 1 - Find the Score

def get_attr_number(node):
    # your code goes here
    score = 0
    for c in node.iter():
        score += int(len(c.attrib))
    return score

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
        
    for c in elem:
        depth(c, level + 1)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(['+91 ' + x[-10:-5] + ' ' + x[-5:] for x in l])
    return fun

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        # complete the function
        return map(f, sorted(people, key = lambda x: int(x[2])))
    return inner

# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#
def birthdayCakeCandles(candles):
    # Write your code here
    i = 0
    max_height = 0
    count = 0
    
    for i in range(len(candles)):
        temp = candles[i]
        if temp > max_height:
            max_height = temp
            count = 1
        elif temp == max_height:
            count += 1
            
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#
def kangaroo(x1, v1, x2, v2):
    # Write your code here
    temp1 = x1 - x2
    temp2 = v2 - v1
    res = ""
    
    if v1 <= v2 and x1 < x2:
        res = "NO"
        return res
        
    elif temp1 % temp2 == 0:
        res = "YES"
        return res
    
    res = "NO"
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
import math
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    total = 0
    k = 5 // 2
    
    for i in range(n):
        total = total + k
        k = 3 * k // 2
    
    return total
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    if len(n) > 1:
        sum_of_digits = 0
        
        for i in range(len(n)):
            sum_of_digits += int(n[i])
            
        temp = str(sum_of_digits * k)
        return superDigit(temp, 1)
        
    return int(n[0])
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    el = arr[n - 1]
    i = len(arr) - 1
    
    while i > 0:
        if el <= arr[i - 1]:
            arr[i] = arr[i - 1]
            print(*arr)
        else:
            arr[i] = el
            print(*arr)
            break
        i -= 1
    
    if i == 0:
        arr[0] = el
        print(*arr)
        
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort2(n, arr):
    # Write your code here
    for i in range(1, n):
        j = i
        while (arr[j] < arr[j-1] and j > 0):
            arr[j-1], arr[j] = arr[j], arr[j-1]
            j -= 1
        else:
            print(*arr)
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

