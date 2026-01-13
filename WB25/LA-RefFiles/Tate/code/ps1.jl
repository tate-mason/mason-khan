"""
Markov Matrix Multiplication From Problem 30 - Intro Linear Algebra Volume 4 by Gilbert Strang
"""

## u matrix

u = [1, 0]
A = [.8 .3;
     .2 .7]

x = u
k = [0:7]

while size(x,2) <= length(k)
  u = A*u
  x = [x u]
end

using Plots
plot(k, x)


# v matrix

v = [0; 1]
A = [.8 .3;
     .2 .7]
x = v
k = [0:7]

for j = 1:7
  v = A*v
  x = [x v]
end
plot(k, x)

# u and v approach steady state s
s = [3/5; 2/5]

s_p = A*s
println(s_p)
