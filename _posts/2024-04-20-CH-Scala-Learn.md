---
title: "关于函数式编程的思考"
date: 2024-04-20 04:35:41 +/-0800
categories: [Programming]
tags: [Scala, CH]     # TAG names should always be lowercase
math: true
image: https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/scala.jpeg
---

近期学校开设了Scala语言的课程，并且老师布置了相关作业，我对这门课学习的质量和动力全靠老师的负责程度，感叹一下，国外老师是真的认真，作业提交均需要在github上建立仓库，并且每次作业都需要创建一个PR，然后老师会进行review，如果通过则合并到主分支，否则提出修改意见，学生再进行修改，如此反复，直到通过为止。学习这门语言给我带了极大的痛苦，因为这门语言的语法和Python、Java、C/C++、JavaScript等语言的语法差异非常大。

## 函数式编程

在敲了多年的Python、C/C++、Java之后，我原以为自己对接触一门新的语言会比较快的上手，但是实际上，我弄错了编程语言和编程范式之间的关系。接触Scala离不开所谓的函数式编程，而函数式编程又离不开所谓的**函数**，理论的东西我也说不清楚，因为我只是非常浅显的入门而已。

## 示例

从示例开始记录属于个人习惯，展示Homework2的有理数实现:

```scala
// `x` and `y` are inaccessible from outside
class Rational(x: Int, y: Int):
// Checking the precondition. Is fails then throws `IllegalArgumentException`
require(y > 0, "Denominator must be positive")

def this(x: Int) = this(x, 1)

val numer = x / g
val denom = y / g

// Defines an external name for a definition
@targetName("less than")
// Annotation on a method definition allows using the method as an infix operation
infix def <(that: Rational): Boolean =
    this.numer * that.denom < that.numer * this.denom

@targetName("less or equal")
infix def <=(that: Rational): Boolean =
    this < that || this == that

@targetName("greater than")
infix def >(that: Rational): Boolean =
    !(this <= that)

@targetName("greater or equal")
infix def >=(that: Rational): Boolean =
    !(this < that)

@targetName("addition")
infix def +(that: Rational): Rational =
    Rational(this.numer * that.denom + that.numer * this.denom, this.denom * that.denom)

@targetName("negation")
infix def unary_- : Rational = Rational(-this.numer, this.denom)

@targetName("substraction")
infix def -(that: Rational): Rational = this + -that

@targetName("multiplication")
infix def *(that: Rational): Rational = Rational(that.numer * this.numer, that.denom * this.denom)

@targetName("devision")
infix def /(that: Rational): Rational = this * that.reciprocal

// Finding the reciprocal of a fraction
def reciprocal: Rational =
    if numer < 0 then Rational(-denom, -numer) else Rational(denom, numer)

override def toString: String = s"${this.numer}/${this.denom}"

@tailrec
private def gcd(a: Int, b: Int): Int =
    if b == 0 then a else gcd(b, a % b)

private lazy val g = gcd(abs(x), y)

override def equals(other: Any): Boolean = other match
    case that: Rational =>
    this.numer * that.denom == that.numer * this.denom
    case _ => false

override def hashCode: Int = 31 * numer + denom

end Rational
```

todo




