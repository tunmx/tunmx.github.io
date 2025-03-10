---
title: "函数式编程-Scala实现整型集合"
date: 2024-04-21 08:51:09 +/-0800
categories: [Programming]
tags: [Scala, CH]     # TAG names should always be lowercase
math: true
image: https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/scala.jpeg
---

## 实现代码

本次作业内容是需要实现一个存放整型数组的集合类(IntSet)，并且需要实现集合的一些基本运算，比如交集（intersection）、并集（union）、差集（difference）、包含（contains）等，并且要实现集合的一些基本操作比如包含（include）、删除（remove）等等。老样子先贴代码。

```scala
object BinaryTree:

def inorderTraversal(set: IntSet): List[Int] = set match
    case Empty => List()
    case NonEmpty(elem, left, right) =>
    inorderTraversal(left) ++ List(elem) ++ inorderTraversal(right)

def listToIntSet(lst: List[Int]): IntSet =
    def buildBalanced(elems: List[Int]): IntSet = elems match
    case Nil => Empty
    case _ =>
        val (left, (mid :: right)) = elems.splitAt(elems.length / 2)
        NonEmpty(mid, buildBalanced(left), buildBalanced(right))

    buildBalanced(lst)

def normalize(set: IntSet): IntSet = listToIntSet(inorderTraversal(set))

abstract class IntSet:

infix def include(x: Int): IntSet

infix def remove(x: Int): IntSet

infix def contains(x: Int): Boolean

@targetName("union")
infix def ∪(that: IntSet): IntSet

@targetName("intersection")
infix def ∩(that: IntSet): IntSet

@targetName("complement")
infix def ∖(that: IntSet): IntSet

@targetName("disjunctive union")
infix def ∆(that: IntSet): IntSet

end IntSet

type Empty = Empty.type

case object Empty extends IntSet:

infix def include(x: Int): IntSet = NonEmpty(x, Empty, Empty)

infix def contains(x: Int): Boolean = false

infix def remove(x: Int): IntSet = this

@targetName("union")
infix def ∪(that: IntSet): IntSet = that

@targetName("intersection")
infix def ∩(that: IntSet): IntSet = this

@targetName("complement")
infix def ∖(that: IntSet): IntSet = this

@targetName("disjunctive union")
infix def ∆(that: IntSet): IntSet = that

override def toString: String = "[*]"


end Empty

case class NonEmpty(elem: Int, left: IntSet, right: IntSet) extends IntSet:

infix def include(x: Int): IntSet =
    BinaryTree.normalize(
    if x < elem       then NonEmpty(elem, left include x, right)
    else if x > elem  then NonEmpty(elem, left, right include x)
    else              this
    )


infix def contains(x: Int): Boolean = 
    if x < elem       then left contains x
    else if x > elem  then right contains x
    else              true

// Optional task
infix def remove(x: Int): IntSet =
    BinaryTree.normalize(
    if x < elem then NonEmpty(elem, left remove x, right)
    else if x > elem then NonEmpty(elem, left, right remove x )
    else left ∪ right
    )

@targetName("union")
infix def ∪(that: IntSet): IntSet = (right ∪ (left ∪ that)) include elem

@targetName("intersection")
infix def ∩(that: IntSet): IntSet =
    if that.contains(elem) then
    (left ∩ that) ∪ (right ∩ that) include elem
    else
    left ∩ that ∪ right ∩ that

@targetName("complement")
infix def ∖(that: IntSet): IntSet =
    if that.contains(elem) then
    (left ∖ that) ∪ (right ∖ that)
    else
    (left ∖ that) ∪ (right ∖ that) include elem

@targetName("disjunctive union")
infix def ∆(that: IntSet): IntSet =
    (this ∖ that) ∪ (that ∖ this)

override def toString: String = s"[$left - [$elem] - $right]"

end NonEmpty
```

## 单元测试

单元测试依旧是老师提供的挖空代码和自己补充得到的：

```scala
// Add additional cases if needed
object EmptySpecification extends Properties("Empty"):
import arbitraries.{given Arbitrary[Int], given Arbitrary[NonEmpty], given Arbitrary[IntSet]}

property("equals to Empty") = propBoolean {
Empty == Empty
}

property("not equal to NonEmpty") = forAll { (nonEmpty: NonEmpty) ⇒
Empty != nonEmpty
}

property("include") = forAll { (element: Int) ⇒
(Empty include element) == NonEmpty(element, Empty, Empty)
}

property("contains") = forAll { (element: Int) ⇒
!(Empty contains element)
}

property("remove") = forAll { (element: Int) ⇒
(Empty remove element) == Empty
}

property("union") = forAll { (set: IntSet) ⇒
(Empty ∪ set) == set
}

property("intersection") = forAll { (set: IntSet) ⇒
(Empty ∩ set) == Empty
}

property("complement of Empty") = forAll { (set: IntSet) ⇒
(set ∖ Empty) == set
}

property("complement of set") = forAll { (set: IntSet) ⇒
(Empty ∖ set) == Empty
}

property("left disjunctive union") = forAll { (set: IntSet) ⇒
(Empty ∆ set) == set
}

property("right disjunctive union") = forAll { (set: IntSet) ⇒
(set ∆ Empty) == set
}

end EmptySpecification

// Add additional cases if needed
object NonEmptySpecification extends Properties("NonEmpty"):
import arbitraries.{given Arbitrary[Int], given Arbitrary[NonEmpty], given Arbitrary[IntSet]}
import arbitraries.{nonEmptyAndAnyElement, removeAnyElement, differentIntSet}
import AbitrariesSpecification.validate

property("normalize") = forAll { (nonEmpty: NonEmpty) ⇒
validate(nonEmpty)
}

property("not equals to Empty") = forAll { (nonEmpty: NonEmpty) ⇒
nonEmpty != Empty
}

property("equal") = forAll { (nonEmpty: NonEmpty) ⇒
nonEmpty == nonEmpty
}

property("not equal") = forAll { (nonEmpty: NonEmpty ) ⇒
forAll(differentIntSet(nonEmpty)) { (different) ⇒
    nonEmpty != different
}
}

property("include") = forAll { (nonEmpty: NonEmpty, element: Int) ⇒
(nonEmpty include element) == listToIntSet(BinaryTree.inorderTraversal(nonEmpty) :+ element)
}

property("contains") = forAll { (nonEmpty: NonEmpty, element: Int) ⇒
(nonEmpty include element) contains element
}

property("remove") = forAll { (nonEmpty: NonEmpty, element: Int) ⇒
// Only one case where the last elem was inserted is removed
!(((nonEmpty include element) remove element) contains element)
}

property("remove any element") = forAll { (nonEmpty: NonEmpty) ⇒
forAll(removeAnyElement(nonEmpty)) { case (setAfterRemoval: IntSet, removedElement: Int) ⇒
    (nonEmpty remove removedElement) == setAfterRemoval
}
}

property("remove and keep BST") = forAll(nonEmptyAndAnyElement) { case (nonEmpty: NonEmpty, element: Int) ⇒
validate(nonEmpty remove element)
}

property("remove a non-existent element") = forAll { (nonEmpty: NonEmpty) ⇒
forAll(removeAnyElement(nonEmpty)) { case (setAfterRemoval: IntSet, removedElement: Int) ⇒
    (setAfterRemoval remove removedElement) == setAfterRemoval
}
}

property("union") = forAll { (nonEmpty: NonEmpty, set: IntSet) ⇒
val unionSet = nonEmpty ∪ set
// It doesn't seem to cover everything ↓
// unionSet ∖ nonEmpty ∖ set  == Empty

val nonEmptyAndSetInUnion = (nonEmpty ∖ unionSet) == Empty && (set ∖ unionSet) == Empty
val noExtraElements = ((unionSet ∖ nonEmpty) ∖ set) == Empty
nonEmptyAndSetInUnion && noExtraElements
}

property("intersection") = forAll { (nonEmpty: NonEmpty, set: IntSet) ⇒
(nonEmpty ∩ set) == (nonEmpty ∪ set) ∖ (nonEmpty ∆ set)
}

property("complement") = forAll { (nonEmpty: NonEmpty, set: IntSet) ⇒
(nonEmpty ∖ set) == (nonEmpty ∪ set) ∆ set
}

property("disjunctive") = forAll { (nonEmpty: NonEmpty, set: IntSet) ⇒
    (nonEmpty ∆ set) == (nonEmpty ∪ set) ∖ (nonEmpty ∩ set)
}

end NonEmptySpecification

// Add additional cases if needed
object IntSetSpecification extends Properties("IntSet"):
import arbitraries.{given Arbitrary[Int], given Arbitrary[IntSet]}
import arbitraries.{removeAnyElement, differentIntSet}
import AbitrariesSpecification.validate

property("normalize") = forAll { (set: IntSet) ⇒
validate(set)
}

property("equals") = forAll { (set: IntSet) ⇒
set == set
}

property("not equal") = forAll { (set: IntSet) ⇒
forAll(differentIntSet(set)) { (different) ⇒
    set != different
}
}

property("include") = forAll { (set: IntSet, element: Int) ⇒
(set include element) == listToIntSet(BinaryTree.inorderTraversal(set) :+ element)
}

property("contains") = forAll { (set: IntSet, element: Int) ⇒
(set include element) contains element
}

property("remove") = forAll { (set: IntSet, element: Int) ⇒
// Only one case where the last elem was inserted is removed
!(((set include element) remove element) contains element)
}

property("remove any element") = forAll { (set: IntSet) =>
forAll(removeAnyElement(set)) { case (setAfterRemoval: IntSet, removedElement: Int) ⇒
    (set remove removedElement) == setAfterRemoval
}
}

property("remove with Empty") = forAll { (set: IntSet, element: Int) ⇒
((set ∖ set) remove element) == Empty
}

property("remove a non-existent element") = forAll { (set: IntSet) ⇒
forAll(removeAnyElement(set)) { case (setAfterRemoval: IntSet, removedElement: Int) ⇒
    (setAfterRemoval remove removedElement) == setAfterRemoval
}
}

property("union") = forAll { (left: IntSet, right: IntSet) ⇒
val unionSet = left ∪ right
val nonEmptyAndSetInUnion = (left ∖ unionSet) == Empty && (right ∖ unionSet) == Empty
val noExtraElements = ((unionSet ∖ left) ∖ right) == Empty
nonEmptyAndSetInUnion && noExtraElements
}

property("intersection") = forAll { (left: IntSet, right: IntSet) ⇒
(left ∩ right) == left ∪ right ∖ (left ∆ right)
}

property("complement") = forAll { (left: IntSet, right: IntSet) ⇒
(left ∖ right) == (left ∪ right) ∆ right
}

property("disjunctive") = forAll { (left: IntSet, right: IntSet) ⇒
(left ∆ right) == (left ∪ right) ∖ (left ∩ right)
}

```

## Equals

除了Equals那部分代码以外，其他单元测试均没太大问题，Equals部分老师给我挖了个坑提了一个要求:

```

That's very good that you followed the contract.
But case objects and case classes have appropriate equals method out of the box, so you don't need to implement it. So I'd rely on that and remove equals and hashCode for Empty and NonEmpty.

```

老师要求的是去掉Empty和NonEmpty的equals和hashCode方法，但是我在实现的时候，发现老师给的代码中，Empty和NonEmpty的equals和hashCode方法并没有实现，所以我就没有实现，而是直接继承了case class和case object的equals和hashCode方法。

但是直接使用这种方法我遇到了一些问题，如果我删除了自定义的equals实现，然后使用case类提供的现成的equals来比较IntSet类的实例，由于排序问题，比较可能会失败，如下面的场景所示：

```scala
// Both sets contain the same number
val setA = listToIntSet(List(5, 2, 19, 4, 7, 12))
val setB = listToIntSet(List(12, 5, 2, 19, 4, 7))
// Will yield False
setA == setB

setA:                           setB:
       5                               12
      / \                             /  \
     /   \                           /    \
    2     19                        5      19
     \   /                         / \
      4 7                        2   7
         \                        \
         12                        4
```

因此，我发现问题从比较两个集合变成了比较两个二叉树。因此，我的方法是实现一个归一化函数，在每次增加或删除节点时，对所有节点重新排序以构建平衡树：

```scala
val normA = BinaryTree.normalize(setA)

setA:                             	normA:
         5                                 7
       /   \                             /   \
      2     19                          4     19
       \   /                           / \   /
        4 7                           2   5 12
           \
           12
```

这样实际上是构成一个二叉搜索树，也叫二叉排序树，这样就可以保证每次比较两个集合时，都能得到正确的结果。

当然这样写是没有效率的，老师也提到了，但是老师也提到了，这个作业的目的是让我们熟悉Scala的函数式编程，课程设置上并没有要求效率，所以我就没有继续优化。




