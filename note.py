#print()的用法：
#选中代码块按alt+shift+e可以单独执行该段代码
#python中的数据类型实际上都是指针，因此赋值操作实际上是将两个指针指向同一个引用地址，更改其中一个，另外一个也会受到影响
#Python 3.x中print之后会自动加一个换行符，Python 2.x则不会
#可以输出数字，字符串和表达式
#None可以用来表示变量为零，函数无有效返回值等等
#包是一种用点式模块名来区分模块的方法
import numpy as np
#数字：
print(520)

#字符串：
print('hello world')

#表达式：
print(3+1)#输出4

#输出到文件：①fp=open('盘符(C/D/E):/文件名加后缀','打开方式') ②print(要写入的东西,file=fp) ③fp.close()

#不进行换行输出：
print('helloworld','helloworld','helloworld')#都写在一行里


#转义字符和c是一样的
print('hello\rworld')  #回车指的是回到本行开头，因此world会把hello覆盖掉
print('hello\bworld')  #\b是退位符，即往后退一格
print('http:\\\\www.baidu.com')

#原字符：不希望转义字符起作用，就在字符串前面加上r/R:比如r'hello\nworld'  f前缀：表示能够通过格式化输出字符串

def saxb(a,x,b): return a*x+b  #定义一个函数

print(saxb(1,2,3))

lambda a,x,b:  a*x+b #匿名函数

print(lambda:(1,2,3))

x=20
print(x+2)
print(x)

y=5
y=y+3
print(y*2)

#Examples:

from urllib.request import urlopen  #表示引进一个urlopen函数，用来爬下url上的数据

#Statement&Expressions:

shakespeare = urlopen('http://composingprograms.com/shakespeare.txt') #urlopen用来打开一个url地址

#Function:

words = set(shakespeare.read().decode().split()) #我们 read url上的数据，将数据decode成text，最后将text split 为words存在set这个类内

#Objects:

print({w for w in words if len(w) == 6 and w[::-1] in words})  #w[::-1]表示逆序遍历w中的每一个字母 in words即表示在words这个类内

print(max(1,2))

#python有很多函数不能在默认情况下直接使用，需要import进一个python library

from math import sqrt
from math import pi
from operator import add,sub,mul

print(sqrt(144))
print(sub(1,2))

#更改一个变量的值不会影响另外一个与其有关联的变量的值

radius = 10
area = pi * radius * radius
print(area)
radius = 11
print(area)

# '='右边的值会先进行计算

x,y=3,4.5
y,x=x,y         #交换x,y的值

#print(...)的返回值为none

print(print(1),print(2))

#Define New Functions:

#def <name> (<formal parameters>):
        #return <return expression>  //return一定要缩进

#对于同一个名称的函数而言，重新定义会覆盖其原先的定义（或值）

#help(函数名)：可以看到函数的docstring (docunmentation：对函数进行说明的文档)

def f(x,y=2):
    return max(x,y)

print(f(1))

def xk(c,d):
    if c==4:
        return 6
    elif d>=4:
        return 6+7+c
    else:
        return 25

def how_big(x):
    if x>10:
        print('huge')
    elif x>5:
        return 'big'
    elif x>0:
        return 'small'
    else:
        print("nothin")

n=3
while n>=0:
    n-=1
    print(n)
#若and中的式子每个均为真，则返回and后面的真值 not就是把所有真改成False(除0,False,None外都是真值)

None and True
True and False
True and 17

def absolute_value(x):
    if x>=0:
        return x
    else:
        return -x

result=absolute_value(-2)

def fib(n):
    """Compute the nth Fibonacci number, for n>=2."""
    pred,curr = 0,1
    k = 2
    while k<n:
        pred,curr=curr,curr+pred  #没有先后顺序，curr赋给pred后，curr+pred中的pred是其原值
        k=k+1
    return curr

assert fib(8) == 13

from doctest import testmod
testmod()


def sum_naturals(n):
    """Return the sum of the first n natural numbers.

    sum_naturals(10)

    sum_naturals(100)
    """
    total,k=0,1
    while k<=n:
        total,k=total+k,k+1
    return total

def count(p,value):
    total,index=0,0
    while index<len(p):
        if p[total]==value:
            total=total+1
        index=index+1
    return total

s=[2,2,7,8]

print(count(s,2))

#for xx in yy:其中yy一定要是一个iterable的数据，比如[[1,2],[2,3]]这样的，也就是list和string

for dnwjkfbas in range(3):
    print("Fuck You!")

#在python的interactive mode中，_用来存储最后一个输出的值

print('C:\some\name')
print(r'C:\some\name')

#下列代码中的\可以用来防止输出空行
print("""\   
Usage:thingy[options]
    -h                         display this usage message
    -H hostname                Hostname to connect to
""")

#下列两个字符直接拼接的代码只能用于字符常量之间，不能用于变量和表达式，变量和表达式直接用+号
'Py'  'thon'

word='Python'
print(word[-1])    #-1是最右边，因为-0和0是一样的，即规定xx[0]为最左边
word[:2]           #[0,2)
word[2:5]          #[2,5)
word[4:]           #[4,end]
word[-2:]          #(end-2,end]

#s[:i]+s[i:]==s恒成立

#python不能更改字符串的值，但可以更改list的值

cubes=[2,6,12,20]

cubes.append(30)

print(cubes)

cubes[1:3]=[]           #清空该段
cubes[:]=[]             #清空该list

a,b=0,1
while a<10:
    print(a,end=' ')             #end可以用于避免空行
    a,b=b,a+b

x = int(input("Please enter an integer: "))
if x<0:
    print("Negative number")
elif x==0:
    print("Zero")
else:
    print("Positive number")

#一边遍历一个collection一边修改它会造成一些误解，可以通过遍历这个collection的copy来解决这个问题,.items()是将字典中的键值对作为一个元组返回
users={'Hans':'activate','京太郎':'activate','马化腾':'inactivate'}
for user,status in users.copy().items():
    if status=='inactivate':
        del users[user]                         #删除整个键值对
active_users={}
for user,status in users.items():
    if status=='activate':
        active_users[user]=status               #对应赋值整个键值对
for user,status in active_users.items():
    print(user,end=' ')

#sum()只能加list、dict这类数据
sum({1,2,3})

#for/while中的else(python中的else不一定和最近的if配对，要看缩进)在loop循环结束时执行
for x in range(2,10):
    for i in range(2,x):
        if x%i==0:
            print(x,'equals',i,'*',x//i)
            break
    else :
        print(x,'is a prime number')

#match和c中的switch很像

def http_error(status):
    match status:
        case 401|402|403:
            return ("Not Found")
        case _:
            return ("Something is wrong with your internet")

print(http_error(404))

from unicodedata import lookup
print(lookup('WHITEHEARTSUIT'))

#raise + 异常名称：输出异常情况

#enum是一个枚举类型：

from enum import Enum

class Color(Enum):
    RED = '1'
    GREEN = 'green'
    BLUE = 'blue'

color = Color(input("Enter your choice of 'red', 'blue' or 'green': "))

match color:
    case Color.RED:
        print("I see red!")
    case Color.GREEN:
        print("Grass is green")
    case Color.BLUE:
        print("I'm feeling the blues :(")


def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')


point = (1, 0)


match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y={y}")
    case (x, 0):
        print(f"X={x}")
    case (x, y):                    # 相当于赋值
        print(f"X={x}, Y={y}")
    case _:
        raise ValueError("Not a Point")

# {}通常用于创建一个字典类型的数据结构，字典类型是一种具有键值对的集合：

my_dict = {"apple":1, "banana":2}

# []用于创建一个元组，元组中的数据可以是python支持的任何一种数据结构

my_list = [1, 2, 3, "apple"]

# 数组形状：比如创建一个形状是（3,1,2）的数组：
a = np.array([[[1, 2], [3, 4], [5, 6]]]) # a有三个二维数组，每个二维数组一行两列
# next()函数的操作对象是一个迭代器而不是可迭代数据，只有经过iter()转换的可迭代数据才可以用next()迭代，iter()并不会改变原数据

# map函数用法：map(a, b)将iterable类型的 b中元素替换为a
squares = list(map(lambda x: x**2, range(10)))
print(squares)

# 输入：input()：用于获取输入的字符串：
age = int(input("input your age: "))
print("Your age is " + str(age)) 