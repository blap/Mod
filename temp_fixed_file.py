"
"
"
\
n
P
e
r
f
from .cuda_wrapper import (
    SM61AttentionWrapper,
    SM61TensorOpsWrapper,
    SM61MemoryPoolWrapper,
    OptimizedAttentionModule,
    OptimizedMLPModule
)
 
v
a
l
i
d
a
t
i
o
n
 
t
e
s
t
 
f
o
r
 
S
M
6
1
 
C
U
D
A
 
k
e
r
n
e
l
s
\
n
T
h
i
s
 
t
e
s
t
 
v
a
l
i
d
a
t
e
s
 
t
h
a
t
 
t
h
e
 
C
U
D
A
 
k
e
r
n
e
l
s
 
p
r
o
v
i
d
e
 
e
x
p
e
c
t
e
d
 
s
p
e
e
d
u
p
 
o
v
e
r
 
C
P
U
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
s
\
n
"
"
"
\
n
i
m
p
o
r
t
 
t
o
r
c
h
\
n
i
m
p
o
r
t
 
t
o
r
c
h
.
n
n
.
f
u
n
c
t
i
o
n
a
l
 
a
s
 
F
\
n
i
m
p
o
r
t
 
t
i
m
e
\
n
i
m
p
o
r
t
 
n
u
m
p
y
 
a
s
 
n
p
\
n
f
r
o
m
 
t
y
p
i
n
g
 
i
m
p
o
r
t
 
T
u
p
l
e
,
 
D
i
c
t
,
 
A
n
y
\
n
\
n
f
r
o
m
 
c
u
d
a
_
k
e
r
n
e
l
s
.
c
u
d
a
_
w
r
a
p
p
e
r
 
i
m
p
o
r
t
 
S
M
6
1
A
t
t
e
n
t
i
o
n
W
r
a
p
p
e
r
,
 
S
M
6
1
T
e
n
s
o
r
O
p
s
W
r
a
p
p
e
r
,
 
S
M
6
1
M
e
m
o
r
y
P
o
o
l
W
r
a
p
p
e
r
,
 
O
p
t
i
m
i
z
e
d
A
t
t
e
n
t
i
o
n
M
o
d
u
l
e
,
 
O
p
t
i
m
i
z
e
d
M
L
P
M
o
d
u
l
e
\
n
 
 
 
 
S
M
6
1
A
t
t
e
n
t
i
o
n
W
r
a
p
p
e
r
,
\
n
 
 
 
 
S
M
6
1
T
e
n
s
o
r
O
p
s
W
r
a
p
p
e
r
,
\
n
 
 
 
 
S
M
6
1
M
e
m
o
r
y
P
o
o
l
W
r
a
p
p
e
r
,
\
n
 
 
 
 
O
p
t
i
m
i
z
e
d
A
t
t
e
n
t
i
o
n
M
o
d
u
l
e
,
\
n
 
 
 
 
O
p
t
i
m
i
z
e
d
M
L
P
M
o
d
u
l
e
\
n
)
\
n
\
n
\
n
d
e
f
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
f
u
n
c
,
 
*
a
r
g
s
,
 
n
u
m
_
r
u
n
s
=
1
0
,
 
w
a
r
m
u
p
=
3
,
 
*
*
k
w
a
r
g
s
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
B
e
n
c
h
m
a
r
k
 
a
 
f
u
n
c
t
i
o
n
 
a
n
d
 
r
e
t
u
r
n
 
t
i
m
i
n
g
 
s
t
a
t
i
s
t
i
c
s
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
#
 
W
a
r
m
u
p
 
r
u
n
s
\
n
 
 
 
 
f
o
r
 
_
 
i
n
 
r
a
n
g
e
(
w
a
r
m
u
p
)
:
\
n
 
 
 
 
 
 
 
 
_
 
=
 
f
u
n
c
(
*
a
r
g
s
,
 
*
*
k
w
a
r
g
s
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
A
c
t
u
a
l
 
t
i
m
i
n
g
 
r
u
n
s
\
n
 
 
 
 
t
i
m
e
s
 
=
 
[
]
\
n
 
 
 
 
f
o
r
 
_
 
i
n
 
r
a
n
g
e
(
n
u
m
_
r
u
n
s
)
:
\
n
 
 
 
 
 
 
 
 
s
t
a
r
t
_
t
i
m
e
 
=
 
t
i
m
e
.
t
i
m
e
(
)
\
n
 
 
 
 
 
 
 
 
_
 
=
 
f
u
n
c
(
*
a
r
g
s
,
 
*
*
k
w
a
r
g
s
)
\
n
 
 
 
 
 
 
 
 
t
o
r
c
h
.
c
u
d
a
.
s
y
n
c
h
r
o
n
i
z
e
(
)
 
 
#
 
E
n
s
u
r
e
 
c
o
m
p
l
e
t
i
o
n
 
f
o
r
 
a
c
c
u
r
a
t
e
 
t
i
m
i
n
g
\
n
 
 
 
 
 
 
 
 
e
n
d
_
t
i
m
e
 
=
 
t
i
m
e
.
t
i
m
e
(
)
\
n
 
 
 
 
 
 
 
 
t
i
m
e
s
.
a
p
p
e
n
d
(
(
e
n
d
_
t
i
m
e
 
-
 
s
t
a
r
t
_
t
i
m
e
)
 
*
 
1
0
0
0
)
 
 
#
 
C
o
n
v
e
r
t
 
t
o
 
m
i
l
l
i
s
e
c
o
n
d
s
\
n
 
 
 
 
\
n
 
 
 
 
r
e
t
u
r
n
 
{
\
n
 
 
 
 
 
 
 
 
'
m
e
a
n
'
:
 
n
p
.
m
e
a
n
(
t
i
m
e
s
)
,
\
n
 
 
 
 
 
 
 
 
'
s
t
d
'
:
 
n
p
.
s
t
d
(
t
i
m
e
s
)
,
\
n
 
 
 
 
 
 
 
 
'
m
i
n
'
:
 
n
p
.
m
i
n
(
t
i
m
e
s
)
,
\
n
 
 
 
 
 
 
 
 
'
m
a
x
'
:
 
n
p
.
m
a
x
(
t
i
m
e
s
)
,
\
n
 
 
 
 
 
 
 
 
'
m
e
d
i
a
n
'
:
 
n
p
.
m
e
d
i
a
n
(
t
i
m
e
s
)
\
n
 
 
 
 
}
\
n
\
n
\
n
d
e
f
 
t
e
s
t
_
a
t
t
e
n
t
i
o
n
_
p
e
r
f
o
r
m
a
n
c
e
(
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
T
e
s
t
 
a
t
t
e
n
t
i
o
n
 
p
e
r
f
o
r
m
a
n
c
e
 
w
i
t
h
 
C
U
D
A
 
o
p
t
i
m
i
z
a
t
i
o
n
s
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
p
r
i
n
t
(
"
T
e
s
t
i
n
g
 
a
t
t
e
n
t
i
o
n
 
p
e
r
f
o
r
m
a
n
c
e
.
.
.
"
)
\
n
 
 
 
 
\
n
 
 
 
 
i
f
 
n
o
t
 
t
o
r
c
h
.
c
u
d
a
.
i
s
_
a
v
a
i
l
a
b
l
e
(
)
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
C
U
D
A
 
n
o
t
 
a
v
a
i
l
a
b
l
e
,
 
s
k
i
p
p
i
n
g
 
p
e
r
f
o
r
m
a
n
c
e
 
t
e
s
t
s
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
t
e
s
t
 
t
e
n
s
o
r
s
\
n
 
 
 
 
b
a
t
c
h
_
s
i
z
e
,
 
s
e
q
_
l
e
n
,
 
n
u
m
_
h
e
a
d
s
,
 
h
e
a
d
_
d
i
m
 
=
 
2
,
 
5
1
2
,
 
8
,
 
6
4
\
n
 
 
 
 
d
e
v
i
c
e
 
=
 
'
c
u
d
a
'
\
n
 
 
 
 
\
n
 
 
 
 
q
u
e
r
y
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
n
u
m
_
h
e
a
d
s
,
 
s
e
q
_
l
e
n
,
 
h
e
a
d
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
k
e
y
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
n
u
m
_
h
e
a
d
s
,
 
s
e
q
_
l
e
n
,
 
h
e
a
d
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
v
a
l
u
e
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
n
u
m
_
h
e
a
d
s
,
 
s
e
q
_
l
e
n
,
 
h
e
a
d
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
C
U
D
A
 
a
t
t
e
n
t
i
o
n
 
w
r
a
p
p
e
r
\
n
 
 
 
 
c
u
d
a
_
a
t
t
e
n
t
i
o
n
 
=
 
S
M
6
1
A
t
t
e
n
t
i
o
n
W
r
a
p
p
e
r
(
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
C
U
D
A
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
\
n
 
 
 
 
c
u
d
a
_
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
c
u
d
a
_
a
t
t
e
n
t
i
o
n
.
f
o
r
w
a
r
d
,
 
q
u
e
r
y
,
 
k
e
y
,
 
v
a
l
u
e
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
P
y
T
o
r
c
h
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
 
f
o
r
 
c
o
m
p
a
r
i
s
o
n
\
n
 
 
 
 
d
e
f
 
p
y
t
o
r
c
h
_
a
t
t
e
n
t
i
o
n
(
q
,
 
k
,
 
v
)
:
\
n
 
 
 
 
 
 
 
 
s
c
o
r
e
s
 
=
 
t
o
r
c
h
.
m
a
t
m
u
l
(
q
,
 
k
.
t
r
a
n
s
p
o
s
e
(
-
2
,
 
-
1
)
)
 
/
 
t
o
r
c
h
.
s
q
r
t
(
t
o
r
c
h
.
t
e
n
s
o
r
(
h
e
a
d
_
d
i
m
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
)
)
\
n
 
 
 
 
 
 
 
 
a
t
t
n
_
w
e
i
g
h
t
s
 
=
 
F
.
s
o
f
t
m
a
x
(
s
c
o
r
e
s
,
 
d
i
m
=
-
1
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
t
o
r
c
h
.
m
a
t
m
u
l
(
a
t
t
n
_
w
e
i
g
h
t
s
,
 
v
)
\
n
 
 
 
 
\
n
 
 
 
 
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
p
y
t
o
r
c
h
_
a
t
t
e
n
t
i
o
n
,
 
q
u
e
r
y
,
 
k
e
y
,
 
v
a
l
u
e
)
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
f
"
C
U
D
A
 
a
t
t
e
n
t
i
o
n
:
 
{
c
u
d
a
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
c
u
d
a
_
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
p
r
i
n
t
(
f
"
P
y
T
o
r
c
h
 
a
t
t
e
n
t
i
o
n
:
 
{
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
\
n
 
 
 
 
s
p
e
e
d
u
p
 
=
 
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
 
/
 
c
u
d
a
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
\
n
 
 
 
 
p
r
i
n
t
(
f
"
S
p
e
e
d
u
p
:
 
{
s
p
e
e
d
u
p
:
.
2
f
}
x
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
V
e
r
i
f
y
 
n
u
m
e
r
i
c
a
l
 
a
c
c
u
r
a
c
y
\
n
 
 
 
 
w
i
t
h
 
t
o
r
c
h
.
n
o
_
g
r
a
d
(
)
:
\
n
 
 
 
 
 
 
 
 
c
u
d
a
_
o
u
t
p
u
t
 
=
 
c
u
d
a
_
a
t
t
e
n
t
i
o
n
.
f
o
r
w
a
r
d
(
q
u
e
r
y
,
 
k
e
y
,
 
v
a
l
u
e
)
\
n
 
 
 
 
 
 
 
 
p
y
t
o
r
c
h
_
o
u
t
p
u
t
 
=
 
p
y
t
o
r
c
h
_
a
t
t
e
n
t
i
o
n
(
q
u
e
r
y
,
 
k
e
y
,
 
v
a
l
u
e
)
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
#
 
C
a
l
c
u
l
a
t
e
 
r
e
l
a
t
i
v
e
 
e
r
r
o
r
\
n
 
 
 
 
 
 
 
 
a
b
s
_
d
i
f
f
 
=
 
t
o
r
c
h
.
a
b
s
(
c
u
d
a
_
o
u
t
p
u
t
 
-
 
p
y
t
o
r
c
h
_
o
u
t
p
u
t
)
\
n
 
 
 
 
 
 
 
 
r
e
l
_
e
r
r
o
r
 
=
 
t
o
r
c
h
.
m
e
a
n
(
a
b
s
_
d
i
f
f
 
/
 
(
t
o
r
c
h
.
a
b
s
(
p
y
t
o
r
c
h
_
o
u
t
p
u
t
)
 
+
 
1
e
-
1
2
)
)
\
n
 
 
 
 
 
 
 
 
m
a
x
_
e
r
r
o
r
 
=
 
t
o
r
c
h
.
m
a
x
(
a
b
s
_
d
i
f
f
)
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
R
e
l
a
t
i
v
e
 
e
r
r
o
r
:
 
{
r
e
l
_
e
r
r
o
r
:
.
6
f
}
"
)
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
M
a
x
 
e
r
r
o
r
:
 
{
m
a
x
_
e
r
r
o
r
:
.
6
f
}
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
P
e
r
f
o
r
m
a
n
c
e
 
s
h
o
u
l
d
 
b
e
 
b
e
t
t
e
r
 
a
n
d
 
n
u
m
e
r
i
c
a
l
 
a
c
c
u
r
a
c
y
 
s
h
o
u
l
d
 
b
e
 
a
c
c
e
p
t
a
b
l
e
\
n
 
 
 
 
a
s
s
e
r
t
 
s
p
e
e
d
u
p
 
>
 
1
.
0
,
 
f
"
E
x
p
e
c
t
e
d
 
s
p
e
e
d
u
p
 
>
 
1
.
0
,
 
g
o
t
 
{
s
p
e
e
d
u
p
}
"
\
n
 
 
 
 
a
s
s
e
r
t
 
r
e
l
_
e
r
r
o
r
 
<
 
1
e
-
4
,
 
f
"
R
e
l
a
t
i
v
e
 
e
r
r
o
r
 
t
o
o
 
h
i
g
h
:
 
{
r
e
l
_
e
r
r
o
r
}
"
\
n
 
 
 
 
a
s
s
e
r
t
 
m
a
x
_
e
r
r
o
r
 
<
 
1
e
-
3
,
 
f
"
M
a
x
 
e
r
r
o
r
 
t
o
o
 
h
i
g
h
:
 
{
m
a
x
_
e
r
r
o
r
}
"
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
"
â
œ
“
 
A
t
t
e
n
t
i
o
n
 
p
e
r
f
o
r
m
a
n
c
e
 
t
e
s
t
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
\
n
\
n
d
e
f
 
t
e
s
t
_
m
a
t
m
u
l
_
p
e
r
f
o
r
m
a
n
c
e
(
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
T
e
s
t
 
m
a
t
r
i
x
 
m
u
l
t
i
p
l
i
c
a
t
i
o
n
 
p
e
r
f
o
r
m
a
n
c
e
 
w
i
t
h
 
C
U
D
A
 
o
p
t
i
m
i
z
a
t
i
o
n
s
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
p
r
i
n
t
(
"
T
e
s
t
i
n
g
 
m
a
t
m
u
l
 
p
e
r
f
o
r
m
a
n
c
e
.
.
.
"
)
\
n
 
 
 
 
\
n
 
 
 
 
i
f
 
n
o
t
 
t
o
r
c
h
.
c
u
d
a
.
i
s
_
a
v
a
i
l
a
b
l
e
(
)
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
C
U
D
A
 
n
o
t
 
a
v
a
i
l
a
b
l
e
,
 
s
k
i
p
p
i
n
g
 
p
e
r
f
o
r
m
a
n
c
e
 
t
e
s
t
s
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
t
e
s
t
 
t
e
n
s
o
r
s
\
n
 
 
 
 
m
,
 
n
,
 
k
 
=
 
1
0
2
4
,
 
1
0
2
4
,
 
1
0
2
4
\
n
 
 
 
 
d
e
v
i
c
e
 
=
 
'
c
u
d
a
'
\
n
 
 
 
 
\
n
 
 
 
 
a
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
m
,
 
k
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
b
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
k
,
 
n
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
t
e
n
s
o
r
 
o
p
s
 
w
r
a
p
p
e
r
\
n
 
 
 
 
t
e
n
s
o
r
_
o
p
s
 
=
 
S
M
6
1
T
e
n
s
o
r
O
p
s
W
r
a
p
p
e
r
(
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
C
U
D
A
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
\
n
 
 
 
 
c
u
d
a
_
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
t
e
n
s
o
r
_
o
p
s
.
m
a
t
m
u
l
,
 
a
,
 
b
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
P
y
T
o
r
c
h
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
 
f
o
r
 
c
o
m
p
a
r
i
s
o
n
\
n
 
 
 
 
d
e
f
 
p
y
t
o
r
c
h
_
m
a
t
m
u
l
(
x
,
 
y
)
:
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
t
o
r
c
h
.
m
a
t
m
u
l
(
x
,
 
y
)
\
n
 
 
 
 
\
n
 
 
 
 
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
p
y
t
o
r
c
h
_
m
a
t
m
u
l
,
 
a
,
 
b
)
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
f
"
C
U
D
A
 
m
a
t
m
u
l
:
 
{
c
u
d
a
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
c
u
d
a
_
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
p
r
i
n
t
(
f
"
P
y
T
o
r
c
h
 
m
a
t
m
u
l
:
 
{
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
\
n
 
 
 
 
s
p
e
e
d
u
p
 
=
 
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
 
/
 
c
u
d
a
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
\
n
 
 
 
 
p
r
i
n
t
(
f
"
S
p
e
e
d
u
p
:
 
{
s
p
e
e
d
u
p
:
.
2
f
}
x
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
V
e
r
i
f
y
 
n
u
m
e
r
i
c
a
l
 
a
c
c
u
r
a
c
y
\
n
 
 
 
 
w
i
t
h
 
t
o
r
c
h
.
n
o
_
g
r
a
d
(
)
:
\
n
 
 
 
 
 
 
 
 
c
u
d
a
_
o
u
t
p
u
t
 
=
 
t
e
n
s
o
r
_
o
p
s
.
m
a
t
m
u
l
(
a
,
 
b
)
\
n
 
 
 
 
 
 
 
 
p
y
t
o
r
c
h
_
o
u
t
p
u
t
 
=
 
p
y
t
o
r
c
h
_
m
a
t
m
u
l
(
a
,
 
b
)
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
#
 
C
a
l
c
u
l
a
t
e
 
r
e
l
a
t
i
v
e
 
e
r
r
o
r
\
n
 
 
 
 
 
 
 
 
a
b
s
_
d
i
f
f
 
=
 
t
o
r
c
h
.
a
b
s
(
c
u
d
a
_
o
u
t
p
u
t
 
-
 
p
y
t
o
r
c
h
_
o
u
t
p
u
t
)
\
n
 
 
 
 
 
 
 
 
r
e
l
_
e
r
r
o
r
 
=
 
t
o
r
c
h
.
m
e
a
n
(
a
b
s
_
d
i
f
f
 
/
 
(
t
o
r
c
h
.
a
b
s
(
p
y
t
o
r
c
h
_
o
u
t
p
u
t
)
 
+
 
1
e
-
1
2
)
)
\
n
 
 
 
 
 
 
 
 
m
a
x
_
e
r
r
o
r
 
=
 
t
o
r
c
h
.
m
a
x
(
a
b
s
_
d
i
f
f
)
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
R
e
l
a
t
i
v
e
 
e
r
r
o
r
:
 
{
r
e
l
_
e
r
r
o
r
:
.
6
f
}
"
)
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
M
a
x
 
e
r
r
o
r
:
 
{
m
a
x
_
e
r
r
o
r
:
.
6
f
}
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
P
e
r
f
o
r
m
a
n
c
e
 
s
h
o
u
l
d
 
b
e
 
b
e
t
t
e
r
 
a
n
d
 
n
u
m
e
r
i
c
a
l
 
a
c
c
u
r
a
c
y
 
s
h
o
u
l
d
 
b
e
 
a
c
c
e
p
t
a
b
l
e
\
n
 
 
 
 
a
s
s
e
r
t
 
s
p
e
e
d
u
p
 
>
 
1
.
0
,
 
f
"
E
x
p
e
c
t
e
d
 
s
p
e
e
d
u
p
 
>
 
1
.
0
,
 
g
o
t
 
{
s
p
e
e
d
u
p
}
"
\
n
 
 
 
 
a
s
s
e
r
t
 
r
e
l
_
e
r
r
o
r
 
<
 
1
e
-
4
,
 
f
"
R
e
l
a
t
i
v
e
 
e
r
r
o
r
 
t
o
o
 
h
i
g
h
:
 
{
r
e
l
_
e
r
r
o
r
}
"
\
n
 
 
 
 
a
s
s
e
r
t
 
m
a
x
_
e
r
r
o
r
 
<
 
1
e
-
3
,
 
f
"
M
a
x
 
e
r
r
o
r
 
t
o
o
 
h
i
g
h
:
 
{
m
a
x
_
e
r
r
o
r
}
"
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
"
â
œ
“
 
M
a
t
m
u
l
 
p
e
r
f
o
r
m
a
n
c
e
 
t
e
s
t
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
\
n
\
n
d
e
f
 
t
e
s
t
_
m
e
m
o
r
y
_
e
f
f
i
c
i
e
n
t
_
o
p
s
_
p
e
r
f
o
r
m
a
n
c
e
(
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
T
e
s
t
 
m
e
m
o
r
y
-
e
f
f
i
c
i
e
n
t
 
o
p
e
r
a
t
i
o
n
s
 
p
e
r
f
o
r
m
a
n
c
e
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
p
r
i
n
t
(
"
T
e
s
t
i
n
g
 
m
e
m
o
r
y
-
e
f
f
i
c
i
e
n
t
 
o
p
e
r
a
t
i
o
n
s
 
p
e
r
f
o
r
m
a
n
c
e
.
.
.
"
)
\
n
 
 
 
 
\
n
 
 
 
 
i
f
 
n
o
t
 
t
o
r
c
h
.
c
u
d
a
.
i
s
_
a
v
a
i
l
a
b
l
e
(
)
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
C
U
D
A
 
n
o
t
 
a
v
a
i
l
a
b
l
e
,
 
s
k
i
p
p
i
n
g
 
p
e
r
f
o
r
m
a
n
c
e
 
t
e
s
t
s
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
t
e
s
t
 
t
e
n
s
o
r
s
\
n
 
 
 
 
b
a
t
c
h
_
s
i
z
e
,
 
s
e
q
_
l
e
n
,
 
h
i
d
d
e
n
_
d
i
m
 
=
 
4
,
 
5
1
2
,
 
1
0
2
4
\
n
 
 
 
 
d
e
v
i
c
e
 
=
 
'
c
u
d
a
'
\
n
 
 
 
 
\
n
 
 
 
 
i
n
p
u
t
_
t
e
n
s
o
r
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
s
e
q
_
l
e
n
,
 
h
i
d
d
e
n
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
w
e
i
g
h
t
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
h
i
d
d
e
n
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
t
e
n
s
o
r
 
o
p
s
 
w
r
a
p
p
e
r
\
n
 
 
 
 
t
e
n
s
o
r
_
o
p
s
 
=
 
S
M
6
1
T
e
n
s
o
r
O
p
s
W
r
a
p
p
e
r
(
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
C
U
D
A
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
 
f
o
r
 
a
d
d
i
t
i
o
n
\
n
 
 
 
 
c
u
d
a
_
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
t
e
n
s
o
r
_
o
p
s
.
m
e
m
o
r
y
_
e
f
f
i
c
i
e
n
t
_
o
p
,
 
i
n
p
u
t
_
t
e
n
s
o
r
,
 
w
e
i
g
h
t
,
 
"
a
d
d
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
P
y
T
o
r
c
h
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
 
f
o
r
 
c
o
m
p
a
r
i
s
o
n
\
n
 
 
 
 
d
e
f
 
p
y
t
o
r
c
h
_
a
d
d
_
o
p
(
x
,
 
w
)
:
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
x
 
+
 
w
 
 
#
 
B
r
o
a
d
c
a
s
t
i
n
g
\
n
 
 
 
 
\
n
 
 
 
 
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
p
y
t
o
r
c
h
_
a
d
d
_
o
p
,
 
i
n
p
u
t
_
t
e
n
s
o
r
,
 
w
e
i
g
h
t
)
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
f
"
C
U
D
A
 
m
e
m
o
r
y
-
e
f
f
i
c
i
e
n
t
 
a
d
d
:
 
{
c
u
d
a
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
c
u
d
a
_
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
p
r
i
n
t
(
f
"
P
y
T
o
r
c
h
 
a
d
d
:
 
{
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
\
n
 
 
 
 
s
p
e
e
d
u
p
 
=
 
p
y
t
o
r
c
h
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
 
/
 
c
u
d
a
_
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
\
n
 
 
 
 
p
r
i
n
t
(
f
"
A
d
d
 
s
p
e
e
d
u
p
:
 
{
s
p
e
e
d
u
p
:
.
2
f
}
x
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
V
e
r
i
f
y
 
n
u
m
e
r
i
c
a
l
 
a
c
c
u
r
a
c
y
 
f
o
r
 
a
d
d
i
t
i
o
n
\
n
 
 
 
 
w
i
t
h
 
t
o
r
c
h
.
n
o
_
g
r
a
d
(
)
:
\
n
 
 
 
 
 
 
 
 
c
u
d
a
_
o
u
t
p
u
t
 
=
 
t
e
n
s
o
r
_
o
p
s
.
m
e
m
o
r
y
_
e
f
f
i
c
i
e
n
t
_
o
p
(
i
n
p
u
t
_
t
e
n
s
o
r
,
 
w
e
i
g
h
t
,
 
"
a
d
d
"
)
\
n
 
 
 
 
 
 
 
 
p
y
t
o
r
c
h
_
o
u
t
p
u
t
 
=
 
p
y
t
o
r
c
h
_
a
d
d
_
o
p
(
i
n
p
u
t
_
t
e
n
s
o
r
,
 
w
e
i
g
h
t
)
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
#
 
C
a
l
c
u
l
a
t
e
 
r
e
l
a
t
i
v
e
 
e
r
r
o
r
\
n
 
 
 
 
 
 
 
 
a
b
s
_
d
i
f
f
 
=
 
t
o
r
c
h
.
a
b
s
(
c
u
d
a
_
o
u
t
p
u
t
 
-
 
p
y
t
o
r
c
h
_
o
u
t
p
u
t
)
\
n
 
 
 
 
 
 
 
 
r
e
l
_
e
r
r
o
r
 
=
 
t
o
r
c
h
.
m
e
a
n
(
a
b
s
_
d
i
f
f
 
/
 
(
t
o
r
c
h
.
a
b
s
(
p
y
t
o
r
c
h
_
o
u
t
p
u
t
)
 
+
 
1
e
-
1
2
)
)
\
n
 
 
 
 
 
 
 
 
m
a
x
_
e
r
r
o
r
 
=
 
t
o
r
c
h
.
m
a
x
(
a
b
s
_
d
i
f
f
)
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
A
d
d
 
r
e
l
a
t
i
v
e
 
e
r
r
o
r
:
 
{
r
e
l
_
e
r
r
o
r
:
.
6
f
}
"
)
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
A
d
d
 
m
a
x
 
e
r
r
o
r
:
 
{
m
a
x
_
e
r
r
o
r
:
.
6
f
}
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
P
e
r
f
o
r
m
a
n
c
e
 
s
h
o
u
l
d
 
b
e
 
b
e
t
t
e
r
 
a
n
d
 
n
u
m
e
r
i
c
a
l
 
a
c
c
u
r
a
c
y
 
s
h
o
u
l
d
 
b
e
 
a
c
c
e
p
t
a
b
l
e
\
n
 
 
 
 
a
s
s
e
r
t
 
s
p
e
e
d
u
p
 
>
 
0
.
5
,
 
f
"
E
x
p
e
c
t
e
d
 
s
p
e
e
d
u
p
 
>
 
0
.
5
,
 
g
o
t
 
{
s
p
e
e
d
u
p
}
"
 
 
#
 
A
l
l
o
w
 
f
o
r
 
s
o
m
e
 
v
a
r
i
a
n
c
e
\
n
 
 
 
 
a
s
s
e
r
t
 
r
e
l
_
e
r
r
o
r
 
<
 
1
e
-
5
,
 
f
"
R
e
l
a
t
i
v
e
 
e
r
r
o
r
 
t
o
o
 
h
i
g
h
:
 
{
r
e
l
_
e
r
r
o
r
}
"
\
n
 
 
 
 
a
s
s
e
r
t
 
m
a
x
_
e
r
r
o
r
 
<
 
1
e
-
6
,
 
f
"
M
a
x
 
e
r
r
o
r
 
t
o
o
 
h
i
g
h
:
 
{
m
a
x
_
e
r
r
o
r
}
"
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
"
â
œ
“
 
M
e
m
o
r
y
-
e
f
f
i
c
i
e
n
t
 
o
p
e
r
a
t
i
o
n
s
 
p
e
r
f
o
r
m
a
n
c
e
 
t
e
s
t
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
\
n
\
n
d
e
f
 
t
e
s
t
_
b
l
o
c
k
_
s
p
a
r
s
e
_
a
t
t
e
n
t
i
o
n
(
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
T
e
s
t
 
b
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
 
f
u
n
c
t
i
o
n
a
l
i
t
y
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
p
r
i
n
t
(
"
T
e
s
t
i
n
g
 
b
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
.
.
.
"
)
\
n
 
 
 
 
\
n
 
 
 
 
i
f
 
n
o
t
 
t
o
r
c
h
.
c
u
d
a
.
i
s
_
a
v
a
i
l
a
b
l
e
(
)
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
C
U
D
A
 
n
o
t
 
a
v
a
i
l
a
b
l
e
,
 
s
k
i
p
p
i
n
g
 
b
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
 
t
e
s
t
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
t
e
s
t
 
t
e
n
s
o
r
s
\
n
 
 
 
 
b
a
t
c
h
_
s
i
z
e
,
 
s
e
q
_
l
e
n
,
 
n
u
m
_
h
e
a
d
s
,
 
h
e
a
d
_
d
i
m
 
=
 
1
,
 
1
2
8
,
 
4
,
 
6
4
\
n
 
 
 
 
d
e
v
i
c
e
 
=
 
'
c
u
d
a
'
\
n
 
 
 
 
\
n
 
 
 
 
q
u
e
r
y
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
n
u
m
_
h
e
a
d
s
,
 
s
e
q
_
l
e
n
,
 
h
e
a
d
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
k
e
y
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
n
u
m
_
h
e
a
d
s
,
 
s
e
q
_
l
e
n
,
 
h
e
a
d
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
v
a
l
u
e
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
n
u
m
_
h
e
a
d
s
,
 
s
e
q
_
l
e
n
,
 
h
e
a
d
_
d
i
m
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
a
 
s
i
m
p
l
e
 
b
l
o
c
k
 
m
a
s
k
 
(
e
v
e
r
y
 
o
t
h
e
r
 
b
l
o
c
k
 
i
s
 
a
c
t
i
v
e
)
\
n
 
 
 
 
b
l
o
c
k
_
s
i
z
e
 
=
 
3
2
\
n
 
 
 
 
n
u
m
_
b
l
o
c
k
s
 
=
 
s
e
q
_
l
e
n
 
/
/
 
b
l
o
c
k
_
s
i
z
e
\
n
 
 
 
 
b
l
o
c
k
_
m
a
s
k
 
=
 
t
o
r
c
h
.
o
n
e
s
(
n
u
m
_
b
l
o
c
k
s
,
 
n
u
m
_
b
l
o
c
k
s
,
 
d
t
y
p
e
=
t
o
r
c
h
.
i
n
t
3
2
,
 
d
e
v
i
c
e
=
d
e
v
i
c
e
)
\
n
 
 
 
 
#
 
M
a
k
e
 
e
v
e
r
y
 
o
t
h
e
r
 
b
l
o
c
k
 
i
n
 
t
h
e
 
u
p
p
e
r
 
t
r
i
a
n
g
u
l
a
r
 
p
a
r
t
 
i
n
a
c
t
i
v
e
 
t
o
 
c
r
e
a
t
e
 
s
p
a
r
s
i
t
y
\
n
 
 
 
 
f
o
r
 
i
 
i
n
 
r
a
n
g
e
(
n
u
m
_
b
l
o
c
k
s
)
:
\
n
 
 
 
 
 
 
 
 
f
o
r
 
j
 
i
n
 
r
a
n
g
e
(
i
+
2
,
 
n
u
m
_
b
l
o
c
k
s
)
:
 
 
#
 
O
n
l
y
 
c
o
m
p
u
t
e
 
l
o
w
e
r
 
t
r
i
a
n
g
u
l
a
r
 
+
 
d
i
a
g
o
n
a
l
 
+
 
o
n
e
 
o
f
f
-
d
i
a
g
o
n
a
l
\
n
 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
(
i
 
+
 
j
)
 
%
 
2
 
=
=
 
0
:
 
 
#
 
A
l
t
e
r
n
a
t
e
 
s
p
a
r
s
i
t
y
 
p
a
t
t
e
r
n
\
n
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
b
l
o
c
k
_
m
a
s
k
[
i
,
 
j
]
 
=
 
0
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
C
U
D
A
 
a
t
t
e
n
t
i
o
n
 
w
r
a
p
p
e
r
 
w
i
t
h
 
b
l
o
c
k
-
s
p
a
r
s
e
 
e
n
a
b
l
e
d
\
n
 
 
 
 
c
u
d
a
_
a
t
t
e
n
t
i
o
n
 
=
 
S
M
6
1
A
t
t
e
n
t
i
o
n
W
r
a
p
p
e
r
(
u
s
e
_
b
l
o
c
k
_
s
p
a
r
s
e
=
T
r
u
e
)
\
n
 
 
 
 
\
n
 
 
 
 
t
r
y
:
\
n
 
 
 
 
 
 
 
 
#
 
T
e
s
t
 
b
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
\
n
 
 
 
 
 
 
 
 
o
u
t
p
u
t
 
=
 
c
u
d
a
_
a
t
t
e
n
t
i
o
n
.
f
o
r
w
a
r
d
(
q
u
e
r
y
,
 
k
e
y
,
 
v
a
l
u
e
,
 
b
l
o
c
k
_
m
a
s
k
=
b
l
o
c
k
_
m
a
s
k
)
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
B
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
 
o
u
t
p
u
t
 
s
h
a
p
e
:
 
{
o
u
t
p
u
t
.
s
h
a
p
e
}
"
)
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
#
 
V
e
r
i
f
y
 
o
u
t
p
u
t
 
s
h
a
p
e
 
i
s
 
c
o
r
r
e
c
t
\
n
 
 
 
 
 
 
 
 
a
s
s
e
r
t
 
o
u
t
p
u
t
.
s
h
a
p
e
 
=
=
 
q
u
e
r
y
.
s
h
a
p
e
,
 
f
"
O
u
t
p
u
t
 
s
h
a
p
e
 
m
i
s
m
a
t
c
h
:
 
{
o
u
t
p
u
t
.
s
h
a
p
e
}
 
v
s
 
{
q
u
e
r
y
.
s
h
a
p
e
}
"
\
n
 
 
 
 
 
 
 
 
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
â
œ
“
 
B
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
 
t
e
s
t
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
e
x
c
e
p
t
 
E
x
c
e
p
t
i
o
n
 
a
s
 
e
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
B
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
 
t
e
s
t
 
f
a
i
l
e
d
 
(
e
x
p
e
c
t
e
d
 
i
f
 
n
o
t
 
f
u
l
l
y
 
i
m
p
l
e
m
e
n
t
e
d
)
:
 
{
e
}
"
)
\
n
 
 
 
 
 
 
 
 
#
 
T
h
i
s
 
i
s
 
e
x
p
e
c
t
e
d
 
i
f
 
t
h
e
 
b
l
o
c
k
-
s
p
a
r
s
e
 
k
e
r
n
e
l
 
i
s
 
n
o
t
 
f
u
l
l
y
 
i
m
p
l
e
m
e
n
t
e
d
\
n
 
 
 
 
 
 
 
 
#
 
T
h
e
 
b
a
s
i
c
 
f
u
n
c
t
i
o
n
a
l
i
t
y
 
s
h
o
u
l
d
 
s
t
i
l
l
 
w
o
r
k
 
w
i
t
h
 
f
a
l
l
b
a
c
k
\
n
 
 
 
 
 
 
 
 
o
u
t
p
u
t
 
=
 
c
u
d
a
_
a
t
t
e
n
t
i
o
n
.
f
o
r
w
a
r
d
(
q
u
e
r
y
,
 
k
e
y
,
 
v
a
l
u
e
)
 
 
#
 
F
a
l
l
b
a
c
k
 
t
o
 
s
t
a
n
d
a
r
d
 
a
t
t
e
n
t
i
o
n
\
n
 
 
 
 
 
 
 
 
a
s
s
e
r
t
 
o
u
t
p
u
t
.
s
h
a
p
e
 
=
=
 
q
u
e
r
y
.
s
h
a
p
e
,
 
f
"
F
a
l
l
b
a
c
k
 
o
u
t
p
u
t
 
s
h
a
p
e
 
m
i
s
m
a
t
c
h
:
 
{
o
u
t
p
u
t
.
s
h
a
p
e
}
 
v
s
 
{
q
u
e
r
y
.
s
h
a
p
e
}
"
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
â
œ
“
 
B
l
o
c
k
-
s
p
a
r
s
e
 
a
t
t
e
n
t
i
o
n
 
f
a
l
l
b
a
c
k
 
t
e
s
t
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
\
n
\
n
d
e
f
 
t
e
s
t
_
m
e
m
o
r
y
_
p
o
o
l
_
p
e
r
f
o
r
m
a
n
c
e
(
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
T
e
s
t
 
m
e
m
o
r
y
 
p
o
o
l
 
p
e
r
f
o
r
m
a
n
c
e
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
p
r
i
n
t
(
"
T
e
s
t
i
n
g
 
m
e
m
o
r
y
 
p
o
o
l
 
p
e
r
f
o
r
m
a
n
c
e
.
.
.
"
)
\
n
 
 
 
 
\
n
 
 
 
 
i
f
 
n
o
t
 
t
o
r
c
h
.
c
u
d
a
.
i
s
_
a
v
a
i
l
a
b
l
e
(
)
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
C
U
D
A
 
n
o
t
 
a
v
a
i
l
a
b
l
e
,
 
s
k
i
p
p
i
n
g
 
m
e
m
o
r
y
 
p
o
o
l
 
t
e
s
t
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
m
e
m
o
r
y
 
p
o
o
l
 
w
r
a
p
p
e
r
\
n
 
 
 
 
m
e
m
o
r
y
_
p
o
o
l
 
=
 
S
M
6
1
M
e
m
o
r
y
P
o
o
l
W
r
a
p
p
e
r
(
p
o
o
l
_
s
i
z
e
=
3
2
 
*
 
1
0
2
4
 
*
 
1
0
2
4
)
 
 
#
 
3
2
M
B
 
p
o
o
l
\
n
 
 
 
 
\
n
 
 
 
 
#
 
T
e
s
t
 
a
l
l
o
c
a
t
i
o
n
 
p
e
r
f
o
r
m
a
n
c
e
\
n
 
 
 
 
a
l
l
o
c
a
t
i
o
n
_
t
i
m
e
s
 
=
 
[
]
\
n
 
 
 
 
f
o
r
 
_
 
i
n
 
r
a
n
g
e
(
1
0
0
)
:
\
n
 
 
 
 
 
 
 
 
s
t
a
r
t
_
t
i
m
e
 
=
 
t
i
m
e
.
t
i
m
e
(
)
\
n
 
 
 
 
 
 
 
 
t
e
n
s
o
r
 
=
 
m
e
m
o
r
y
_
p
o
o
l
.
a
l
l
o
c
a
t
e
_
t
e
n
s
o
r
(
(
1
0
2
4
,
 
1
0
2
4
)
,
 
d
t
y
p
e
=
t
o
r
c
h
.
f
l
o
a
t
3
2
)
\
n
 
 
 
 
 
 
 
 
t
o
r
c
h
.
c
u
d
a
.
s
y
n
c
h
r
o
n
i
z
e
(
)
\
n
 
 
 
 
 
 
 
 
e
n
d
_
t
i
m
e
 
=
 
t
i
m
e
.
t
i
m
e
(
)
\
n
 
 
 
 
 
 
 
 
a
l
l
o
c
a
t
i
o
n
_
t
i
m
e
s
.
a
p
p
e
n
d
(
(
e
n
d
_
t
i
m
e
 
-
 
s
t
a
r
t
_
t
i
m
e
)
 
*
 
1
0
0
0
)
 
 
#
 
m
s
\
n
 
 
 
 
\
n
 
 
 
 
a
v
g
_
a
l
l
o
c
a
t
i
o
n
_
t
i
m
e
 
=
 
n
p
.
m
e
a
n
(
a
l
l
o
c
a
t
i
o
n
_
t
i
m
e
s
)
\
n
 
 
 
 
p
r
i
n
t
(
f
"
A
v
e
r
a
g
e
 
a
l
l
o
c
a
t
i
o
n
 
t
i
m
e
:
 
{
a
v
g
_
a
l
l
o
c
a
t
i
o
n
_
t
i
m
e
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
h
e
c
k
 
s
t
a
t
s
\
n
 
 
 
 
s
t
a
t
s
 
=
 
m
e
m
o
r
y
_
p
o
o
l
.
g
e
t
_
s
t
a
t
s
(
)
\
n
 
 
 
 
p
r
i
n
t
(
f
"
M
e
m
o
r
y
 
p
o
o
l
 
s
t
a
t
s
:
 
{
s
t
a
t
s
}
"
)
\
n
 
 
 
 
\
n
 
 
 
 
a
s
s
e
r
t
 
a
v
g
_
a
l
l
o
c
a
t
i
o
n
_
t
i
m
e
 
<
 
1
0
.
0
,
 
f
"
A
l
l
o
c
a
t
i
o
n
 
t
i
m
e
 
t
o
o
 
h
i
g
h
:
 
{
a
v
g
_
a
l
l
o
c
a
t
i
o
n
_
t
i
m
e
}
m
s
"
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
"
â
œ
“
 
M
e
m
o
r
y
 
p
o
o
l
 
p
e
r
f
o
r
m
a
n
c
e
 
t
e
s
t
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
\
n
\
n
d
e
f
 
t
e
s
t
_
f
u
l
l
_
m
o
d
e
l
_
i
n
t
e
g
r
a
t
i
o
n
(
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
T
e
s
t
 
f
u
l
l
 
m
o
d
e
l
 
i
n
t
e
g
r
a
t
i
o
n
 
w
i
t
h
 
C
U
D
A
 
o
p
t
i
m
i
z
a
t
i
o
n
s
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
p
r
i
n
t
(
"
T
e
s
t
i
n
g
 
f
u
l
l
 
m
o
d
e
l
 
i
n
t
e
g
r
a
t
i
o
n
.
.
.
"
)
\
n
 
 
 
 
\
n
 
 
 
 
i
f
 
n
o
t
 
t
o
r
c
h
.
c
u
d
a
.
i
s
_
a
v
a
i
l
a
b
l
e
(
)
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
C
U
D
A
 
n
o
t
 
a
v
a
i
l
a
b
l
e
,
 
s
k
i
p
p
i
n
g
 
m
o
d
e
l
 
i
n
t
e
g
r
a
t
i
o
n
 
t
e
s
t
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
\
n
 
 
 
 
#
 
C
r
e
a
t
e
 
a
 
s
m
a
l
l
 
c
o
n
f
i
g
 
f
o
r
 
t
e
s
t
i
n
g
\
n
 
 
 
 
c
l
a
s
s
 
T
e
s
t
C
o
n
f
i
g
:
\
n
 
 
 
 
 
 
 
 
h
i
d
d
e
n
_
s
i
z
e
 
=
 
2
5
6
\
n
 
 
 
 
 
 
 
 
n
u
m
_
a
t
t
e
n
t
i
o
n
_
h
e
a
d
s
 
=
 
8
\
n
 
 
 
 
 
 
 
 
n
u
m
_
h
i
d
d
e
n
_
l
a
y
e
r
s
 
=
 
2
\
n
 
 
 
 
 
 
 
 
i
n
t
e
r
m
e
d
i
a
t
e
_
s
i
z
e
 
=
 
5
1
2
\
n
 
 
 
 
 
 
 
 
h
i
d
d
e
n
_
a
c
t
 
=
 
"
s
i
l
u
"
\
n
 
 
 
 
 
 
 
 
h
i
d
d
e
n
_
d
r
o
p
o
u
t
_
p
r
o
b
 
=
 
0
.
0
\
n
 
 
 
 
 
 
 
 
a
t
t
e
n
t
i
o
n
_
d
r
o
p
o
u
t
_
p
r
o
b
 
=
 
0
.
0
\
n
 
 
 
 
 
 
 
 
m
a
x
_
p
o
s
i
t
i
o
n
_
e
m
b
e
d
d
i
n
g
s
 
=
 
5
1
2
\
n
 
 
 
 
 
 
 
 
i
n
i
t
i
a
l
i
z
e
r
_
r
a
n
g
e
 
=
 
0
.
0
2
\
n
 
 
 
 
 
 
 
 
l
a
y
e
r
_
n
o
r
m
_
e
p
s
 
=
 
1
e
-
6
\
n
 
 
 
 
 
 
 
 
p
a
d
_
t
o
k
e
n
_
i
d
 
=
 
0
\
n
 
 
 
 
 
 
 
 
v
o
c
a
b
_
s
i
z
e
 
=
 
1
0
0
0
\
n
 
 
 
 
 
 
 
 
u
s
e
_
c
a
c
h
e
 
=
 
T
r
u
e
\
n
 
 
 
 
 
 
 
 
n
u
m
_
k
e
y
_
v
a
l
u
e
_
h
e
a
d
s
 
=
 
N
o
n
e
\
n
 
 
 
 
\
n
 
 
 
 
c
o
n
f
i
g
 
=
 
T
e
s
t
C
o
n
f
i
g
(
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
T
e
s
t
 
o
p
t
i
m
i
z
e
d
 
a
t
t
e
n
t
i
o
n
 
m
o
d
u
l
e
\
n
 
 
 
 
a
t
t
e
n
t
i
o
n
_
m
o
d
u
l
e
 
=
 
O
p
t
i
m
i
z
e
d
A
t
t
e
n
t
i
o
n
M
o
d
u
l
e
(
c
o
n
f
i
g
)
.
c
u
d
a
(
)
\
n
 
 
 
 
\
n
 
 
 
 
b
a
t
c
h
_
s
i
z
e
,
 
s
e
q
_
l
e
n
 
=
 
2
,
 
6
4
\
n
 
 
 
 
h
i
d
d
e
n
_
s
t
a
t
e
s
 
=
 
t
o
r
c
h
.
r
a
n
d
n
(
b
a
t
c
h
_
s
i
z
e
,
 
s
e
q
_
l
e
n
,
 
c
o
n
f
i
g
.
h
i
d
d
e
n
_
s
i
z
e
,
 
d
e
v
i
c
e
=
'
c
u
d
a
'
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
t
h
e
 
a
t
t
e
n
t
i
o
n
 
m
o
d
u
l
e
\
n
 
 
 
 
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
a
t
t
e
n
t
i
o
n
_
m
o
d
u
l
e
.
f
o
r
w
a
r
d
,
 
h
i
d
d
e
n
_
s
t
a
t
e
s
)
\
n
 
 
 
 
p
r
i
n
t
(
f
"
O
p
t
i
m
i
z
e
d
 
a
t
t
e
n
t
i
o
n
 
m
o
d
u
l
e
:
 
{
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
T
e
s
t
 
o
p
t
i
m
i
z
e
d
 
M
L
P
 
m
o
d
u
l
e
\
n
 
 
 
 
m
l
p
_
m
o
d
u
l
e
 
=
 
O
p
t
i
m
i
z
e
d
M
L
P
M
o
d
u
l
e
(
c
o
n
f
i
g
)
.
c
u
d
a
(
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
B
e
n
c
h
m
a
r
k
 
t
h
e
 
M
L
P
 
m
o
d
u
l
e
\
n
 
 
 
 
r
e
s
u
l
t
s
 
=
 
b
e
n
c
h
m
a
r
k
_
f
u
n
c
t
i
o
n
(
m
l
p
_
m
o
d
u
l
e
.
f
o
r
w
a
r
d
,
 
h
i
d
d
e
n
_
s
t
a
t
e
s
)
\
n
 
 
 
 
p
r
i
n
t
(
f
"
O
p
t
i
m
i
z
e
d
 
M
L
P
 
m
o
d
u
l
e
:
 
{
r
e
s
u
l
t
s
[
'
m
e
a
n
'
]
:
.
3
f
}
m
s
 
Â
±
 
{
r
e
s
u
l
t
s
[
'
s
t
d
'
]
:
.
3
f
}
m
s
"
)
\
n
 
 
 
 
\
n
 
 
 
 
#
 
T
e
s
t
 
f
o
r
w
a
r
d
 
p
a
s
s
\
n
 
 
 
 
w
i
t
h
 
t
o
r
c
h
.
n
o
_
g
r
a
d
(
)
:
\
n
 
 
 
 
 
 
 
 
a
t
t
n
_
o
u
t
p
u
t
 
=
 
a
t
t
e
n
t
i
o
n
_
m
o
d
u
l
e
(
h
i
d
d
e
n
_
s
t
a
t
e
s
)
\
n
 
 
 
 
 
 
 
 
m
l
p
_
o
u
t
p
u
t
 
=
 
m
l
p
_
m
o
d
u
l
e
(
h
i
d
d
e
n
_
s
t
a
t
e
s
)
\
n
 
 
 
 
\
n
 
 
 
 
a
s
s
e
r
t
 
a
t
t
n
_
o
u
t
p
u
t
[
0
]
.
s
h
a
p
e
 
=
=
 
h
i
d
d
e
n
_
s
t
a
t
e
s
.
s
h
a
p
e
,
 
f
"
A
t
t
e
n
t
i
o
n
 
o
u
t
p
u
t
 
s
h
a
p
e
 
m
i
s
m
a
t
c
h
"
\
n
 
 
 
 
a
s
s
e
r
t
 
m
l
p
_
o
u
t
p
u
t
.
s
h
a
p
e
 
=
=
 
h
i
d
d
e
n
_
s
t
a
t
e
s
.
s
h
a
p
e
,
 
f
"
M
L
P
 
o
u
t
p
u
t
 
s
h
a
p
e
 
m
i
s
m
a
t
c
h
"
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
"
â
œ
“
 
F
u
l
l
 
m
o
d
e
l
 
i
n
t
e
g
r
a
t
i
o
n
 
t
e
s
t
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
\
n
\
n
d
e
f
 
r
u
n
_
p
e
r
f
o
r
m
a
n
c
e
_
v
a
l
i
d
a
t
i
o
n
(
)
:
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
R
u
n
 
a
l
l
 
p
e
r
f
o
r
m
a
n
c
e
 
v
a
l
i
d
a
t
i
o
n
 
t
e
s
t
s
\
n
 
 
 
 
"
"
"
\
n
 
 
 
 
p
r
i
n
t
(
"
R
u
n
n
i
n
g
 
p
e
r
f
o
r
m
a
n
c
e
 
v
a
l
i
d
a
t
i
o
n
 
t
e
s
t
s
 
f
o
r
 
S
M
6
1
 
C
U
D
A
 
k
e
r
n
e
l
s
.
.
.
\
n
"
)
\
n
 
 
 
 
\
n
 
 
 
 
t
e
s
t
s
 
=
 
[
\
n
 
 
 
 
 
 
 
 
t
e
s
t
_
a
t
t
e
n
t
i
o
n
_
p
e
r
f
o
r
m
a
n
c
e
,
\
n
 
 
 
 
 
 
 
 
t
e
s
t
_
m
a
t
m
u
l
_
p
e
r
f
o
r
m
a
n
c
e
,
\
n
 
 
 
 
 
 
 
 
t
e
s
t
_
m
e
m
o
r
y
_
e
f
f
i
c
i
e
n
t
_
o
p
s
_
p
e
r
f
o
r
m
a
n
c
e
,
\
n
 
 
 
 
 
 
 
 
t
e
s
t
_
b
l
o
c
k
_
s
p
a
r
s
e
_
a
t
t
e
n
t
i
o
n
,
\
n
 
 
 
 
 
 
 
 
t
e
s
t
_
m
e
m
o
r
y
_
p
o
o
l
_
p
e
r
f
o
r
m
a
n
c
e
,
\
n
 
 
 
 
 
 
 
 
t
e
s
t
_
f
u
l
l
_
m
o
d
e
l
_
i
n
t
e
g
r
a
t
i
o
n
,
\
n
 
 
 
 
]
\
n
 
 
 
 
\
n
 
 
 
 
p
a
s
s
e
d
 
=
 
0
\
n
 
 
 
 
t
o
t
a
l
 
=
 
l
e
n
(
t
e
s
t
s
)
\
n
 
 
 
 
\
n
 
 
 
 
f
o
r
 
t
e
s
t
_
f
u
n
c
 
i
n
 
t
e
s
t
s
:
\
n
 
 
 
 
 
 
 
 
t
r
y
:
\
n
 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
t
e
s
t
_
f
u
n
c
(
)
:
\
n
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
a
s
s
e
d
 
+
=
 
1
\
n
 
 
 
 
 
 
 
 
e
x
c
e
p
t
 
E
x
c
e
p
t
i
o
n
 
a
s
 
e
:
\
n
 
 
 
 
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
T
e
s
t
 
{
t
e
s
t
_
f
u
n
c
.
_
_
n
a
m
e
_
_
}
 
f
a
i
l
e
d
 
w
i
t
h
 
e
r
r
o
r
:
 
{
e
}
"
)
\
n
 
 
 
 
 
 
 
 
 
 
 
 
i
m
p
o
r
t
 
t
r
a
c
e
b
a
c
k
\
n
 
 
 
 
 
 
 
 
 
 
 
 
t
r
a
c
e
b
a
c
k
.
p
r
i
n
t
_
e
x
c
(
)
\
n
 
 
 
 
\
n
 
 
 
 
p
r
i
n
t
(
f
"
\
n
P
e
r
f
o
r
m
a
n
c
e
 
v
a
l
i
d
a
t
i
o
n
 
r
e
s
u
l
t
s
:
 
{
p
a
s
s
e
d
}
/
{
t
o
t
a
l
}
 
t
e
s
t
s
 
p
a
s
s
e
d
"
)
\
n
 
 
 
 
\
n
 
 
 
 
i
f
 
p
a
s
s
e
d
 
=
=
 
t
o
t
a
l
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
ð
Ÿ
Ž
‰
 
A
l
l
 
p
e
r
f
o
r
m
a
n
c
e
 
v
a
l
i
d
a
t
i
o
n
 
t
e
s
t
s
 
p
a
s
s
e
d
!
"
)
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
â
œ
…
 
C
U
D
A
 
k
e
r
n
e
l
s
 
p
r
o
v
i
d
e
 
e
x
p
e
c
t
e
d
 
s
p
e
e
d
u
p
 
o
v
e
r
 
C
P
U
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
s
"
)
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
â
œ
…
 
N
u
m
e
r
i
c
a
l
 
a
c
c
u
r
a
c
y
 
i
s
 
m
a
i
n
t
a
i
n
e
d
"
)
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
"
â
œ
…
 
A
l
l
 
c
o
m
p
o
n
e
n
t
s
 
a
r
e
 
p
r
o
p
e
r
l
y
 
i
n
t
e
g
r
a
t
e
d
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e
\
n
 
 
 
 
e
l
s
e
:
\
n
 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
â

Œ
 
{
t
o
t
a
l
 
-
 
p
a
s
s
e
d
}
 
t
e
s
t
s
 
f
a
i
l
e
d
"
)
\
n
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
F
a
l
s
e
\
n
\
n
\
n
i
f
 
_
_
n
a
m
e
_
_
 
=
=
 
"
_
_
m
a
i
n
_
_
"
:
\
n
 
 
 
 
s
u
c
c
e
s
s
 
=
 
r
u
n
_
p
e
r
f
o
r
m
a
n
c
e
_
v
a
l
i
d
a
t
i
o
n
(
)
\
n
 
 
 
 
i
f
 
n
o
t
 
s
u
c
c
e
s
s
:
\
n
 
 
 
 
 
 
 
 
e
x
i
t
(
1
)
