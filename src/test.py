def fibonacci(n):
    """
    Generate the first n numbers of the Fibonacci sequence.

    Parameters:
    n (int): The number of Fibonacci numbers to generate.

    Returns:
    list: A list containing the first n numbers of the Fibonacci sequence.
    """
    fib_sequence = []
    a, b = 0, 1
    for i in range(n):
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence


if __name__ == "__main__":
    print(fibonacci(10))
