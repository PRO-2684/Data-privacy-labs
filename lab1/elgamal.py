import random
import numpy as np
import sympy

DEFAULT_KEY_SIZE = 256


def is_primitive_root(g: int, p: int, factors: dict) -> bool:
    # determine whether g is a primitive root of p
    for factor in factors:
        if pow(g, (p - 1) // factor, p) == 1:
            return False
    return True


def generate_p_and_g(n_bit: int) -> (int, int):
    while True:
        # generate an n-bit random prime number p
        p = sympy.randprime(2 ** (n_bit - 1), 2**n_bit)

        # compute the prime factorization of p-1
        factors = sympy.factorint(p - 1).keys()

        # choose a possible primitive root g
        for g in range(2, p):
            if is_primitive_root(g, p, factors):
                return p, g


def mod_exp(base: int, exponent: int, modulus: int) -> int:
    """DONE: calculate (base^exponent) mod modulus.
    Recommend to use the fast power algorithm.
    """
    result = pow(base, exponent, modulus)
    return result


def elgamal_key_generation(key_size: int) -> ((int, int, int), int):
    """Generate the keys based on the key_size."""
    # generate a large prime number p and a primitive root g
    p, g = generate_p_and_g(key_size)

    # FIXME: generate private key x and public key y here.
    x = random.randint(1, p - 2)
    y = mod_exp(g, x, p)

    return (p, g, y), x


def elgamal_encrypt(public_key: (int, int, int), plaintext: int) -> (int, int):
    """FIXME: encrypt the plaintext with the public key."""
    p, g, y = public_key
    # generate a random number k (temporary private key such that 1 <= k <= p-2)
    k = random.randint(1, p - 2)
    # compute temporary public key c1
    c1 = mod_exp(g, k, p)
    # compute encrypted message c2
    c2 = plaintext * mod_exp(y, k, p) % p
    return c1, c2


def elgamal_decrypt(public_key: (int, int, int), private_key: int, ciphertext: (int, int)) -> int:
    """FIXME: decrypt the ciphertext with the public key and the private key."""
    p, g, y = public_key
    c1, c2 = ciphertext
    # compute the inverse of c1 s
    s = mod_exp(c1, private_key, p)
    # compute the inverse of s s_
    s_ = sympy.core.numbers.mod_inverse(s, p)
    # compute the plaintext
    plaintext = c2 * s_ % p
    return plaintext


if __name__ == "__main__":
    # set key_size, such as 256, 1024...
    key_size = input("Please input the key size: ")
    key_size = int(key_size) if key_size else DEFAULT_KEY_SIZE

    # generate keys
    public_key, private_key = elgamal_key_generation(key_size)
    print("Public Key:", public_key)
    print("Private Key:", private_key)

    # encrypt plaintext
    plaintext = int(input("Please input an integer: "))
    ciphertext = elgamal_encrypt(public_key, plaintext)
    print("Ciphertext:", ciphertext)

    # decrypt ciphertext
    decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
    print("Decrypted Text:", decrypted_text)
