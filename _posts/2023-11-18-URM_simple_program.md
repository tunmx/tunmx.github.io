---
title: "Unlimited Register Machine: Simple Program Implementation"
date: 2023-11-18 13:15:35 +/-0800
categories: [Programming]
tags: [URM, Computability Theory, EN]     # TAG names should always be lowercase
math: true
image: https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/v2-377e01e9b8a5e7a5ee856473555ec0c7_720w.webp
---

## 0. Intro

In the previously studied theory of computability, I learned that the Unlimited Register Machine (URM) is a theoretical computational model, consisting of infinitely many registers and a set of basic instructions, used to simulate any computable process.

To verify the theories I have learned, I will use Python to implement a simple URM (Unlimited Register Machine) model manually, and utilize it to execute some basic computational logic and cases.

## 1. Infinite Register Machine

An infinite register machine, commonly referred to as URM, is a theoretical device characterized by the following properties:

### 1.1. Registers

An URM is equipped with an array of registers that can contain natural numbers: $$N = \{0, 1, 2, \ldots\}$$.

Any specific URM program might use only a limited number of these registers, but the model does not impose an upper limit.

These registers are denoted as $${R_{0}, R_{1}, R_{2}, R_{3}, \ldots}$$.

The value held at any given moment in a register is represented by the corresponding lowercase letter $$r_{0}, r_{1}, r_{2}, r_{3}, \ldots$$.

The registers are unlimited in two senses:
1. While an URM program might employ a finite number of registers, there is no fixed maximum number that a particular URM program can use.
2. There is no limit to the magnitude of natural numbers that can be stored in any of the registers.

### 1.2. Program

The values stored in a URM's registers are modified following a specific set of rules defined by a program. This program in a URM is comprised of a finite series of fundamental instructions.

### 1.3. Operation

During the operation of a URM, the initial step involves executing the program's first instruction. After completing an instruction, the URM proceeds to the subsequent instruction for execution, except when a Jump instruction dictates a different course of action.

### 1.4. Instruction Pointer

The instruction pointer, often referred to as IP, is a conceptual marker in a URM that indicates the line number of the instruction set to be executed next. It functions akin to a specialized register within the URM, specifically designated to store this line number.

### 1.5. Stage of Computation

The computational stage, or simply stage, of a URM program reflects the total number of instructions that have been executed up to that point. Consequently, each stage is equivalent to the execution of a single instruction.

### 1.6. State

The state of a URM program at any given stage is characterized by the following elements:

- The current value of the instruction pointer (IP).
- The values held in all the registers that the URM program has utilized up to that stage.

### 1.7. Input

The input to a URM program is:
either an ordered k-tuple (x1, x2, ..., xk) in N^k
or a natural number x in N.
In the latter case, it is convenient to consider a single natural number as an ordered 1-tuple (x1) in N^1 = N. Hence we can discuss inputs to URM programs solely as instances of tuples, and not be concerned with cumbersome repetition for the cases where k = 1 and otherwise.

The convention usually used is for a URM program P to start computation with:
the input (x1, x2, ..., xk) in registers R1, R2, ..., Rk and 0 in all other registers used by P.
That is, the initial state of the URM is: ∀i ∈ [1..k], ri = xi; and ∀i > k, ri = 0 ∧ r0 = 0.

It is usual for the input (either all or part) to be overwritten during the course of the operation of a program. That is, at the end of a program, R1, R2, ..., Rk are not guaranteed still to contain x1, x2, ..., xk unless the program has been explicitly written so as to ensure that this is the case.

### 1.8. Output

At the end of the running of a URM program, the output will be found in register $$R_{0}$$.

### 1.9. Null Program

The null URM program is a URM program which contains no instructions. That is, a URM program whose length is zero.

### 1.10. Instructions

| Name      | Code   | Effect               | Explanation                                                    |
|-----------|--------|----------------------|----------------------------------------------------------------|
| Zero      | Z(n)   | 0 → $$R_n$$;             | Put $$0$$ in register $$R_n$$                                          |
|           |        | $$i_p$$ + 1 → $$IP$$         | and pass to the next instruction.                              |
| Successor | S(n)   | $$r_n$$ + 1 → $$R_n$$;       | Increase register $$R_n $$                                         |
|           |        | $$i_p$$ + 1 → $$IP$$         | by 1 and pass to the next instruction.                         |
| Copy      | C(m,n) | $$r_m$$ → $$R_n$$;           | Copy the content of register $$R_m$$                               |
|           |        | $$i_p$$ + 1 → $$IP$$         | to register $$R_n$$ and pass to the next instruction.              |
| Jump      | J(m,n,q)| $$q$$ → $$IP$$ if r_m = $$r_n$$ | If the contents of $$R_m$$ and $$R_n$$ are the same,                   |
|           |        | else $$i_p$$ + 1 → $$IP$$    | then pass to the instruction numbered $$q$$,                       |
|           |        |                      | otherwise pass to the next instruction.                        |

## 2. How to implement it using Python.

In Python, I developed a URM (Unlimited Register Machine) simulator to validate the accuracy of my instructions. Key components of this implementation include several classes such as Instructions, Registers, URMSimulator, along with foundational operators for URM and various functions like size, haddr, normalize, concat, and reloc. However, these implementations are rudimentary and might contain some errors.

### 2.1. Instructions
Instructions represents a list of URM (Unlimited Register Machine) instructions.

This class is used to store and manipulate a sequence of instructions for a URM simulator.
Each instruction in the list is a tuple representing a specific operation in the URM.

```python
class Instructions(object):
    def __init__(self, *inst, ):
        """
        Initializes the Instructions object.

        :param inst: A list of tuples where each tuple represents a URM instruction.
                     Each instruction is a tuple consisting of an operation code followed
                     by its arguments. If None is provided, initializes with an empty list.
        """
        self.instructions = list()
        for item in inst:
            if isinstance(item, list):
                self.instructions += item
            elif isinstance(item, Instructions):
                self.instructions += item.instructions
            elif isinstance(item, tuple):
                self.instructions.append(item)
            else:
                raise TypeError("Input data error.")

    def __str__(self):
        return str(self.instructions)

    def copy(self):
        return copy.deepcopy(self)

    def summary(self):
        # Define the format for each row of the table
        row_format = "{:<5}\t{:<3}\t{:<4}\t{:<4}\t{:<7}\n"
        # Create a table header with lines
        header_line = '-' * 40
        table = row_format.format("Line", "Op", "Arg1", "Arg2", "Jump To") + header_line + '\n'

        # Iterate over the instructions and their indices
        for index, instruction in enumerate(self.instructions, 1):
            # Prepare args with empty strings for missing values
            args = [''] * 3
            # Fill the args list with actual values from the instruction
            for i, arg in enumerate(instruction[1:]):
                args[i] = str(arg)
            # Format each line as a row in the table
            line = row_format.format(index, instruction[0], *args)
            table += line

        return table

    def __getitem__(self, index):
        return self.instructions[index]

    def __setitem__(self, index, value):
        if not isinstance(value, tuple):
            raise ValueError("Type error.")
        self.instructions[index] = value

    def __iter__(self):
        return iter(self.instructions)

    def __len__(self):
        return len(self.instructions)

    def __add__(self, other):
        if not isinstance(other, Instructions):
            raise ValueError("Operand must be an instance of Instructions.")

        # Use concatenation function
        return Instructions.concatenation(self, other)

    def append(self, item):
        if isinstance(item, list):
            self.instructions += item
        elif isinstance(item, Instructions):
            self.instructions += item.instructions
        else:
            self.instructions.append(item)

    def haddr(self):
        highest_register = -1
        for instruction in self.instructions:
            op = instruction[0]
            if op in ['Z', 'S']:  # These instructions involve only one register
                highest_register = max(highest_register, instruction[1])
            elif op in ['C', 'J']:  # These instructions involving two registers
                highest_register = max(highest_register, instruction[1], instruction[2])
        return highest_register if highest_register >= 0 else None

    @staticmethod
    def normalize(instructions):
        normalized_instructions = Instructions()
        for instruction in instructions:
            if instruction[0] == 'J':  # Check if it's a jump instruction
                m, n, k = instruction[1], instruction[2], instruction[3]
                if not 1 <= k <= len(instructions):  # Check if k is out of range
                    k = len(instructions) + 1  # Set k to n + 1
                normalized_instructions.append(('J', m, n, k))
            else:
                normalized_instructions.append(instruction)

```

### 2.2. Registers

Registers represents a list of register values for a URM (Unlimited Register Machine).
This class is used to store the state of the registers in a URM simulator.
Each register can hold an integer value. The registers are indexed starting from 0.

```python
class Registers(object):
    def __init__(self, lis: List[int]):
        """
        Initializes the Registers object with a given list of integers.

        Each integer in the list represents the initial value of a register in the URM.
        The registers are indexed in the order they appear in the list.

        :param lis: A list of integers representing the initial values of the registers.
                    Each integer must be non-negative, as URM registers can only hold
                    natural numbers (including zero).

        :raises ValueError: If any item in the list is not an integer or is a negative integer.
        """
        for item in lis:
            if not isinstance(item, int):
                raise ValueError("All items in the list must be integers")
            if item < 0:
                raise ValueError("An integer greater than 0 must be entered")

        self.registers = lis

    def copy(self):
        return copy.deepcopy(self)

    def summary(self):
        headers = [f"R{i}" for i in range(len(self.registers))]
        divider = '-' * (len(headers) * 8 - 1)
        header_row = '\t'.join(headers)
        values_row = '\t'.join(map(str, self.registers))
        table = f"{header_row}\n{divider}\n{values_row}"
        return table

    def __str__(self):
        return str(self.registers)

    def __getitem__(self, index):
        return self.registers[index]

    def __setitem__(self, index, value):
        if not isinstance(value, int):
            raise ValueError("Only integers can be assigned")
        if value < 0:
            raise ValueError("An integer greater than 0 must be entered")
        self.registers[index] = value

    def __len__(self):
        return len(self.registers)

    @staticmethod
    def allocate(num: int):
        r = [0 for _ in range(num)]
        reg = Registers(r)
        return reg

```

### 2.3. URM Simulator

Implementation scheme for simulating an Unlimited Register Machine, realizing the computational logic of four types of instructions: zero, successor, copy, and jump.

```python
@dataclass
class URMResult(object):
    num_of_steps: int
    ops_of_steps: List[str]
    registers_of_steps: List[Registers]
    last_registers: Registers

class URMSimulator(object):
    @staticmethod
    def _execute_zero(registers: Registers, n: int) -> Registers:
        """
        Set the value of register number n to 0.
        """
        registers[n] = 0
        return registers

    @staticmethod
    def _execute_successor(registers: Registers, n: int) -> Registers:
        """
        Increment the value of register number n.
        """
        registers[n] += 1
        return registers

    @staticmethod
    def _execute_copy(registers: Registers, j: int, k: int) -> Registers:
        """
        Copy the value of register number j to register number k.
        """
        registers[k] = registers[j]
        return registers

    @staticmethod
    def _execute_jump(registers: Registers, m: int, n: int, q: int, current_line: int) -> int:
        """
        Jump to line 'q' if values in registers 'm' and 'n' are equal, else go to the next line.
        """
        if registers[m] == registers[n]:
            return q - 1  # Adjust for zero-based indexing
        else:
            return current_line + 1

    @staticmethod
    def execute_instructions(instructions: Instructions, initial_registers: Registers,
                             safety_count: int = 1000) -> Generator:
        """
        Execute a set of URM (Unlimited Register Machine) instructions.

        :param instructions: The set of URM instructions to execute.
        :param initial_registers: The initial state of the registers.
        :param safety_count: Maximum number of iterations to prevent infinite loops.
        :return: Generator yielding the state of the registers after each instruction.
        """
        registers = initial_registers
        exec_instructions = copy.deepcopy(instructions)
        exec_instructions.append(('END',))
        current_line = 0
        count = 0

        while current_line < len(exec_instructions):
            if count > safety_count:
                raise ValueError("The number of cycles exceeded the safe number.")

            instruction = exec_instructions[current_line]
            op = instruction[0]

            try:
                if op == 'Z':
                    registers = URMSimulator._execute_zero(registers, instruction[1])
                    current_line += 1
                elif op == 'S':
                    registers = URMSimulator._execute_successor(registers, instruction[1])
                    current_line += 1
                elif op == 'C':
                    registers = URMSimulator._execute_copy(registers, instruction[1], instruction[2])
                    current_line += 1
                elif op == 'J':
                    jump_result = URMSimulator._execute_jump(registers, instruction[1], instruction[2], instruction[3],
                                                             current_line)
                    current_line = jump_result if jump_result != -1 else len(exec_instructions)
                elif op == 'END':
                    break
                count += 1
            except Exception as e:
                raise RuntimeError(f"Error executing instruction at line {current_line}: {e}")

            # print(registers, instruction)
            yield copy.deepcopy(registers), f"{current_line}: {op}" + "(" + ", ".join(map(str, instruction[1:])) + ")"

    @staticmethod
    def forward(param: Dict[int, int], initial_registers: Registers, instructions: Instructions,
                safety_count: int = 1000) -> URMResult:
        registers = copy.deepcopy(initial_registers)
        if not isinstance(param, dict):
            raise TypeError("Input param must be a dictionary")
        for key, value in param.items():
            if not isinstance(key, int):
                raise TypeError("All keys must be integers")
            if not isinstance(value, int):
                raise TypeError("All values must be integers")
            if value < 0:
                raise ValueError("Input Value must be a natural number")
            if key < 0:
                raise ValueError("Input Index must be a natural number")
            registers[key] = value

        registers_list = [copy.deepcopy(registers), ]
        ops_info = ['Initial']
        if len(registers) < instructions.haddr():
            raise ValueError("The number of registers requested cannot satisfy this set of instructions.")
        gen = URMSimulator.execute_instructions(instructions=instructions, initial_registers=registers,
                                                safety_count=safety_count)
        num_of_steps = 0
        last_registers = None
        for registers_moment, command in gen:
            num_of_steps += 1
            ops_info.append(command)
            registers_list.append(registers_moment)
            last_registers = registers_moment
        result = URMResult(ops_of_steps=ops_info, registers_of_steps=registers_list, last_registers=last_registers,
                           num_of_steps=num_of_steps)

        return result

```

### 2.4. Interface Design

We have designed some general interfaces for the URM simulator to more easily build URM instructions and simulate computational results.

```python
def urm_op(func):
    """
    Decorator to convert the function to op.
    """

    def wrapper(*args):
        function_name = func.__name__
        return (function_name, *args)

    return wrapper


@urm_op
def C():
    """
    URM Copy operation. Copies the value from one register to another.
    """
    pass


@urm_op
def J():
    """
    URM Jump operation. Jumps to a specified line if two registers hold the same value.
    """
    pass


@urm_op
def Z():
    """
    URM Zero operation. Sets the value of a register to zero.
    """
    pass


@urm_op
def S():
    """
    URM Successor operation. Increments the value of a register by one.
    """
    pass


_END = "END"  # Marker for the end of a URM program (used internally)


def size(instructions: Instructions) -> int:
    """
    Calculates the number of instructions in a URM program.

    :param instructions: An Instructions object representing a URM program.
    :return: The number of instructions in the program.
    """
    return len(instructions)


def haddr(instructions: Instructions) -> int:
    """
    Finds the highest register address used in a URM program.

    :param instructions: An Instructions object representing a URM program.
    :return: The highest register index used in the program.
    """
    return instructions.haddr()


def normalize(instructions: Instructions) -> Instructions:
    """
    Normalizes a URM program so that all jump operations target valid instruction lines.

    :param instructions: An Instructions object representing a URM program.
    :return: A new Instructions object with normalized jump targets.
    """
    return Instructions.normalize(instructions)


def concat(p: Instructions, q: Instructions) -> Instructions:
    """
    Concatenates two URM programs into a single program.

    :param p: An Instructions object representing the first URM program.
    :param q: An Instructions object representing the second URM program.
    :return: A new Instructions object with the concatenated program.
    """
    return Instructions.concatenation(p, q)


def reloc(instructions: Instructions, alloc: Tuple[int, ...]) -> Instructions:
    """
    Relocates the register addresses in a URM program according to a specified mapping.

    :param instructions: An Instructions object representing a URM program.
    :param alloc: A tuple defining the new register addresses for each original address.
    :return: A new Instructions object with relocated register addresses.
    """
    return Instructions.relocation(instructions, alloc)


def allocate(num: int) -> Registers:
    """
    Allocates a specified number of registers, initializing them with zero values.

    This function creates a new Registers object with a given number of registers.
    Each register is initialized with the value 0.

    :param num: The number of registers to allocate.
    :return: A Registers object with 'num' registers, each initialized to 0.
    """
    return Registers.allocate(num)


@cost("URM program")
def forward(param: Dict[int, int], initial_registers: Registers, instructions: Instructions,
            safety_count: int = 1000) -> URMResult:
    """
    Executes a URM (Unlimited Register Machine) simulation with given parameters, initial registers, and instructions.

    This function sets up the registers according to the input parameters, then runs the URM simulation
    with the provided instructions. It executes the instructions step by step and keeps track of the
    state of the registers after each step, returning the final result of the simulation.

    :param param: A dictionary representing the input parameters for the URM simulation.
                  The keys are register indices (int), and the values are the initial values (int) for those registers.
    :param initial_registers: A Registers object representing the initial state of all registers.
                              This object is modified during the simulation according to the URM instructions.
    :param instructions: An Instructions object representing the set of URM instructions to be executed.
    :param safety_count: An integer specifying the maximum number of steps to simulate.
                         This prevents infinite loops in the simulation.

    :return: An URMResult object that contains information about the simulation,
             including the number of steps executed, the operations performed in each step,
             the state of the registers after each step, and the final state of the registers.

    Raises:
        AssertionError: If the input parameters are not a dictionary with integer keys and values,
                        or if the initial values are not non-negative integers,
                        or if the number of registers is insufficient for the given instructions.
    """
    return URMSimulator.forward(param=param, initial_registers=initial_registers, instructions=instructions,
                                safety_count=safety_count)
```


### 2.5. Example-1: sum(x, y): 

We have designed a URM program to perform the addition of two numbers: $$sum(x, y) = x + y$$, following the convention that $$R[0]$$ is for output, and $$R[1]$$ and $$R[3]$$ are for parameters. The maximum number of registers used in this program is 3. The program is as follows:

```python
from urm_simulation import *

# sum(x, y) program instructions set
sum_instructions = Instructions(
    C(2, 0),
    Z(2),
    J(1, 2, 0),
    S(0),
    S(2),
    J(0, 0, 3)
)

P = sum_instructions
num_of_registers = haddr(P) + 1
registers = allocate(num_of_registers)
print("Init Registers: ")
print(registers.summary())

x, y = 5, 9
param = {1: x, 2: y}

# Run program
result = forward(param, registers, P, safety_count=100000)

for idx, reg in enumerate(result.registers_of_steps):
    command = result.ops_of_steps[idx]
    print(f"[step {idx}] {command}")
    print(reg.summary())
    print("")


```

Output after executing the program:

```
Init Registers: 
R0      R1      R2
-----------------------
0       0       0

[step 0] Initial
R0      R1      R2
-----------------------
0       5       9

[step 1] 1: C(2, 0)
R0      R1      R2
-----------------------
9       5       9

[step 2] 2: Z(2)
R0      R1      R2
-----------------------
9       5       0

[step 3] 3: J(1, 2, 0)
R0      R1      R2
-----------------------
9       5       0

[step 4] 4: S(0)
R0      R1      R2
-----------------------
10      5       0

[step 5] 5: S(2)
R0      R1      R2
-----------------------
10      5       1

[step 6] 2: J(0, 0, 3)
R0      R1      R2
-----------------------
10      5       1

[step 7] 3: J(1, 2, 0)
R0      R1      R2
-----------------------
10      5       1

[step 8] 4: S(0)
R0      R1      R2
-----------------------
11      5       1

[step 9] 5: S(2)
R0      R1      R2
-----------------------
11      5       2

[step 10] 2: J(0, 0, 3)
R0      R1      R2
-----------------------
11      5       2

[step 11] 3: J(1, 2, 0)
R0      R1      R2
-----------------------
11      5       2

[step 12] 4: S(0)
R0      R1      R2
-----------------------
12      5       2

[step 13] 5: S(2)
R0      R1      R2
-----------------------
12      5       3

[step 14] 2: J(0, 0, 3)
R0      R1      R2
-----------------------
12      5       3

[step 15] 3: J(1, 2, 0)
R0      R1      R2
-----------------------
12      5       3

[step 16] 4: S(0)
R0      R1      R2
-----------------------
13      5       3

[step 17] 5: S(2)
R0      R1      R2
-----------------------
13      5       4

[step 18] 2: J(0, 0, 3)
R0      R1      R2
-----------------------
13      5       4

[step 19] 3: J(1, 2, 0)
R0      R1      R2
-----------------------
13      5       4

[step 20] 4: S(0)
R0      R1      R2
-----------------------
14      5       4

[step 21] 5: S(2)
R0      R1      R2
-----------------------
14      5       5

[step 22] 2: J(0, 0, 3)
R0      R1      R2
-----------------------
14      5       5

[step 23] 7: J(1, 2, 0)
R0      R1      R2
-----------------------
14      5       5
```

This is a simple case that helps us understand the common techniques used in URM (Unlimited Register Machine). By employing the method of accumulation, we implement the logic of sum.


### 2.6. Example-2: fibb(): 


For this example, we will use the URM instruction set to design a URM program capable of implementing the Fibonacci function.

The Fibonacci sequence is defined by a simple recursive formula:

$$Fibb(n) = Fibb(n-1) + Fibb(n-2)$$,

where F(n) represents the nth term of the sequence. The sequence starts with:

$$Fibb(0) = 0$$ 
; 
$$Fibb(1) = 1$$.


This means the sequence begins with 0 and 1, and each subsequent term is the sum of the two preceding ones. For example, the first few terms of the Fibonacci sequence are 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, and so on.

```python
from urm_simulation import *

# sum(x, y) program instructions set
fibb_instructions = Instructions(
        J(1, 0, 0),  # 1. if R1 == 0 then fibb(0) = 0
        S(0),  # 2. set R0 = 2
        J(1, 0, 0),  # 3. if R1 = 1 then fibb(1) = 1
        S(2),  # 4. set R2 = 1

        J(1, 2, 0),  # 5. loop_1 if R1 == R2 then jump to the end

        S(2),  # 6. set k++
        C(0, 4),  # 7. set fibb(k-1) to R4
        Z(0),  # 8. set R0 to 0
        Z(5),  # 9. set R5 to 0

        C(4, 0),  # 10. copy R4 to 0
        J(5, 3, 15),  # 11. if R5 == R3 then jump to 15
        S(0),  # 12. set R0++
        S(5),  # 13. set R5++
        J(1, 1, 11),  # 14. do while
        C(4, 3),  # 15. copy fibb(k-1) for the current k to fibb(k-2) for the next k(k++)

        J(2, 2, 5),  # 16 do while
)

P = fibb_instructions
num_of_registers = haddr(P) + 1
registers = allocate(num_of_registers)
print("Init Registers: ")
print(registers.summary())

x = 10
param = {1: x,}

# Run program
result = forward(param, registers, P, safety_count=100000)

last = result.last_registers
print(last.summary())

```


Output after executing the program:

```
Init Registers: 
R0      R1      R2      R3      R4      R5
-----------------------------------------------
0       0       0       0       0       0

URM program@Cost of forward(): 0.009 seconds
R0      R1      R2      R3      R4      R5
-----------------------------------------------
55      10      10      34      34      21

```


This case verifies the result of the URM program's execution, in the scenario where fibb(10)=55.

## 3. Install the URM simulator

If you want to use the URM simulator, you can install it using the following command:

```bash
pip install urm
```

## 4. Conclusion


The above documents my implementation of a URM simulator using Python, along with experiments on several simple cases.