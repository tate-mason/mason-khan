using LinearAlgebra, Statistics, Plots, LaTeXStrings

#"""
#This file serves as a companion for lesson 2 of "QuantEcon: Julia" course
#"""
#
#randn() # Creates a random number from a normal distribution N(0,1)
#
#n = 100
#ε = randn(n) # Creates a vector of 100 random numbers from N(0,1)
#plot(1:n, ε) # Plots the random numbers
#
## Arrays
#typeof(ε) # Check the type of ε
#ε[1:5] # Access the first 5 elements of ε
#
## For Loops 
#
#n = 100
#ep = zeros(n)
#for i in 1:n
#  ep[i] = randn()
#end
#
#"""
#first, declare ep as a vector of n numbers initialized to zero
#then, loop over the integers from 1 to n
#at each iteration, assign a random normal number to the i-th position of ep
#"""
#
##better style
#n = 100
#ep = zeros(n)
#for i in eachindex(ep)
#  ep[i] = randn()
#end
#"""
#Using eachindex is better style as it works for arrays of any dimension
#"""
#
##ep_sum = 0.0
##m = 5
##for i in 1:m
##  ep_sum += ep_val[i]
##end
##ep_mean = ep_sum / m
#"""
#This code computes the mean of the first m elements of ep_val
#"""
#
## Functions
#function gen_dat(n)
#  ep = zeros(n)
#  for i in eachindex(ep)
#    ep[i] = (randn())^2
#  end
#  return ep
#end
#
#dat = gen_dat(10)
#plot(dat)
#
#"""
#This function generates a vector of n random numbers from a chi-squared distribution with 1 degree of freedom
#"""
#
## Better style
#function gen_dat2(n)
#  ep = randn(n)
#  return ep .^ 2
#end
#dat2 = gen_dat2(10)
#
## Broadcasting
#f(x) = x^2
#gen_dat(n) = f.(randn(n))
#dat = gen_dat(10)
#
#"""
#This code defines a function f that squares its input, then generates n random normal numbers and applies f to each element using broadcasting
#"""
#
## Useful functions
#using Distributions
#
#function plot_hist(distribution, n)
#  ep = rand(distribution, n)
#  histogram(ep)
#end
#lp = Laplace()
#plot_hist(lp, 1000)
#
#"""
#Generate a histogram of 1000 random numbers from the laplace distribution.
#lp is bound to the same distribution as distribution in the function.
#"""
#
## Fixed Point Maps
#
#p = 1.0
#beta = 0.9
#maxiter = 1000
#tol = 1e-7
#v_iv = 0.8 # initial guess
#
#v_old = v_iv
#normdiff = Inf
#iter = 1
#
#while normdiff > tol && iter <= maxiter
#  v_new = p + beta * v_old
#  normdiff = norm(v_new - v_old)
#
#  v_old = v_new
#  iter += 1
#end
#
#println("Fixed point = $v_old
#      |f(x) - x| = $normdiff in $iter iterations")
#
### Check at each iteration
#v_old = v_iv
#normdiff = Inf
#iter = 1
#
#for i in 1:maxiter
#  v_new = p + beta * v_old
#  normdiff = norm(v_new - v_old)
#  if normdiff < tol
#    iter = i
#    break
#  end
#  v_old = v_new
#end
#println("Fixed point = $v_old
#      |f(x) - x| = $normdiff in $iter iterations")
#
## Passing a Function
#
#function fpm(f, iv, tol, maxiter)
#  x_old = iv
#  normdiff = Inf
#  iter = 1
#  while normdiff > tol && iter <= maxiter
#    x_new = f(x_old)
#    normdiff = norm(x_new - x_old)
#    x_old = x_new
#    iter += 1
#  end
#  return (x_old, normdiff, iter)
#end
#
#p = 1.0
#beta = 0.9
#f(v) = p + beta * v
#
#maxiter = 1000
#tol = 1e-7
#v_init = 0.8
#
#v_star, normdiff, iter = fpm(f, v_init, tol, maxiter)
#
#println("Fixed point = $v_star
#      |f(x) - x| = $normdiff in $iter iterations")
#
## Named arguments and Return Values
#
#function fpm(f, iv; tol = 1e-7, maxiter = 1000)
#  x_old = iv
#  normdiff = Inf
#  iter = 1
#  while normdiff > tol && iter <= maxiter
#    x_new = f(x_old) # use the passed in map
#    normdiff = norm(x_new - x_old)
#    x_old = x_new
#    iter += 1
#  end
#  return (; value = x_old, nromdiff, iter) # return a named tuple
#end
#
#p = 1.0
#beta = 0.9
#f(v) = p + beta * v
#
#sol = fpm(f, 0.8; tol = 1e-10) # don't need to pass maxiter
#println("Fixed point = $(sol.value)
#      |f(x) - x| = $(sol.normdiff) in $(sol.iter) iterations")

# Exercises
# 1. Write a factorial function using a for loop.

function factorial2(n)
  x = 1
  for i in 2:n
    x *= i
  end
  return x
end

fac = factorial2(10)
println("x = ", fac)

# 2. Binomial Number Drawer

function binomial_rv(n, p)
  s = 0
  for i in 1:n
    s += rand() < p
  end
  return s
end

Y = binomial_rv(10, 0.3)
println(Y)
