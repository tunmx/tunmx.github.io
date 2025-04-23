---
title: "关于函数式编程入门示例"
date: 2024-04-20 04:35:41 +/-0800
categories: [Programming]
tags: [Scala, CH]     # TAG names should always be lowercase
math: true
image: https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/scala.jpeg
---

近期学校开设了Scala语言的课程，并且老师布置了相关作业，我对这门课学习的质量和动力全靠老师的负责程度，感叹一下，国外老师是真的认真，作业提交均需要在github上建立仓库，并且每次作业都需要创建一个PR，然后老师会进行review，如果通过则合并到主分支，否则提出修改意见，学生再进行修改，如此反复，直到通过为止。学习这门语言给我带了极大的痛苦，因为这门语言的语法和Python、Java、C/C++、JavaScript等语言的语法差异非常大。

## 函数式编程

在敲了多年的Python、C/C++、Java之后，我原以为自己对接触一门新的语言会比较快的上手，但是实际上，我弄错了编程语言和编程范式之间的关系。接触Scala离不开所谓的函数式编程，而函数式编程又离不开所谓的**函数**，理论的东西我也说不清楚，因为我只是非常浅显的入门而已。

## 示例

从示例开始记录属于个人习惯，直接展示Homework2的有理数实现作为示例。

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

这个示例主要目的是使用Scala的语法特性实现一个有理数类，并且实现有理数之间的运算。

其中既展示了Scala的一些语法，也初步展示了函数式编程的一些基础概念。

### 不可变性（Immutability）

Rational类使用不可变值（val numer，val denom）而不是可变状态的变量。一旦创建了有理数对象，其分子和分母就不会再改变。

### 纯函数（Pure Functions）

方法如+、-、*和/不会修改现有对象，而是返回新的Rational实例。例如：

- r1 + r2创建一个新的Rational，而不是改变r1或r2
- -的实现重用了+和一元否定，展示了纯函数的组合

### 惰性求值（Lazy Evaluation）

private lazy val g展示了惰性求值——最大公约数只在首次需要时计算，而不是在对象创建时。这在该值不总是被使用的情况下可以提高性能。

### 尾递归（Tail Recursion）

gcd方法使用尾递归（标记为@tailrec）而不是迭代循环，这在函数式编程中很常见。Scala编译器会优化尾递归函数以避免栈溢出。

### 模式匹配（Pattern Matching）

equals方法使用模式匹配（other match case that: Rational => ...）来处理不同类型，这是函数式编程的基本构造。

### 运算符重载（Operator Overloading）

代码使用Scala定义运算符作为方法的能力，使有理数行为像内置数值类型，并允许以自然符号书写数学表达式。运算符重载的能力在其他语言也很常见，如C++、Python等都支持。

## 单元测试

单元测试是软件开发中非常重要的一部分，Scala的单元测试框架也非常成熟，这里使用ScalaTest进行单元测试。

```scala
import arbitraries.{given Arbitrary[Int], given Arbitrary[Rational], given Arbitrary[Integer]}

property("throw exception due to zero denominator") = forAll { (numer: Int) ⇒
throws(classOf[IllegalArgumentException]) {
    Rational(numer, 0)
}
}

property("throw exception due to negative denominator") = forAll { (numer: Int, kindaDenom: Int) ⇒
throws(classOf[IllegalArgumentException]) {
    Rational(numer, -abs(kindaDenom))
}
}

property("check that rational number is simplified") = forAll { (numer: Int, int: Int) ⇒
val denom = abs(int) + 1
val rational = Rational(numer, denom)

rational.numer == (numer / gcd(abs(numer), denom)) && rational.denom == (denom / gcd(abs(numer), denom))
}

property("check equals") = forAll { (left: Rational, right: Rational) ⇒
(left == right) == (left.numer == right.numer && left.denom == right.denom)
}

property("less then") = forAll { (left: Rational, right: Rational) =>
(left < right) == (left.numer * right.denom < right.numer * left.denom)
}

property("less or equal") = forAll { (left: Rational, right: Rational) =>
(left <= right) == ( left < right || left == right)
}

property("greater") = forAll { (left: Rational, right: Rational) =>
(left > right) == !(left <= right)
}

property("greater or equal") = forAll { (left: Rational, right: Rational) =>
(left >= right) == ( left > right || left == right)
}

property("negation") = forAll { (rational: Rational) =>
-rational == Rational(-rational.numer, rational.denom)
}

property("addition") = forAll { (left: Rational, right: Rational) =>
left + right == Rational(
    left.numer * right.denom + right.numer * left.denom,
    left.denom * right.denom
)
}

property("subtraction") = forAll { (left: Rational, right: Rational) =>
left - right == Rational(
    left.numer * right.denom - right.numer * left.denom,
    left.denom * right.denom
)
}

property("multiplication") = forAll { (left: Rational, right: Rational) =>
left * right == Rational(
    left.numer * right.numer,
    left.denom * right.denom
)
}

property("division by positive") = forAll { (left: Rational, numer: PositiveInteger, denom: PositiveInteger) =>
val right = Rational(numer, denom)
left / right == Rational(left.numer * right.denom, left.denom * right.numer)
}

property("division by negative") = forAll { (left: Rational, numer: NegativeInteger, denom: PositiveInteger) =>
val right = Rational(numer, denom)
val sign = signum(left.denom * right.numer)
left / right == Rational(left.numer * right.denom * sign, abs(left.denom * right.numer))
}

property("division by zero") = forAll { (left: Rational, int: PositiveInteger) =>
throws(classOf[IllegalArgumentException]) {
    left / Rational(0, int)
}
}


```

以上单元测试的内容属于被老师挖空后，我根据要求补全的，使用ScalaTest的forAll方法进行单元测试，也是修改了好几遍才达到要求，老师要求是Scala需要满足函数式编程的优雅特性，还需要保持Scala代码风格简约，说实话写这样的代码对于其他面向过程和面向对象的语言来说，确实有点困难，需要转变思维。


