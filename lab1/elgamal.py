import random
import sympy

DEFAULT_KEY_SIZE = 256


def is_primitive_root(g: int, p: int, factors: dict) -> bool:
    """Determine whether g is a primitive root of p"""
    for factor in factors:
        if pow(g, (p - 1) // factor, p) == 1:
            return False
    return True


def generate_p_and_g(n_bit: int) -> (int, int):
    """Generate a large prime number p and its primitive root g."""
    while True:
        p = sympy.randprime(
            2 ** (n_bit - 1), 2**n_bit
        )  # Generate an n-bit random prime number p
        factors = sympy.factorint(
            p - 1
        ).keys()  # Compute the prime factorization of p-1
        # Choose a possible primitive root g
        for g in range(2, p):
            if is_primitive_root(g, p, factors):
                return p, g  # Found primitive root g


def mod_exp(base: int, exponent: int, modulus: int) -> int:
    """Calculate (base^exponent) mod modulus using the fast power algorithm."""
    result = pow(base, exponent, modulus)
    return result


def elgamal_key_generation(key_size: int) -> ((int, int, int), int):
    """Generate the keys based on the key_size."""
    p, g = generate_p_and_g(
        key_size
    )  # Generate a large prime number p and a primitive root g

    # Generate private key x and public key y.
    x = random.randint(1, p - 2)
    y = mod_exp(g, x, p)

    return (p, g, y), x


def elgamal_encrypt(public_key: (int, int, int), plaintext: int) -> (int, int):
    """Encrypt the plaintext with the public key."""
    p, g, y = public_key
    k = random.randint(1, p - 2)  # Generate a temporary private key k (1 <= k <= p-2)
    c1 = mod_exp(g, k, p)  # Compute temporary public key c1
    c2 = plaintext * mod_exp(y, k, p) % p  # Compute encrypted message c2
    return c1, c2


def elgamal_decrypt(
    public_key: (int, int, int), private_key: int, ciphertext: (int, int)
) -> int:
    """DONE: decrypt the ciphertext with the public key and the private key."""
    p, g, y = public_key
    c1, c2 = ciphertext
    s = mod_exp(c1, private_key, p)  # Compute the inverse of c1 s
    s_ = sympy.core.numbers.mod_inverse(s, p)  # Compute the inverse of s, s_
    plaintext = c2 * s_ % p  # Compute the plaintext
    return plaintext


if __name__ == "__main__":
    # Set key_size, such as 256, 1024...
    key_size = input("Please input the key size: ")
    key_size = int(key_size) if key_size else DEFAULT_KEY_SIZE

    # Generate keys
    public_key, private_key = elgamal_key_generation(key_size)
    print("Public Key:", public_key)
    print("Private Key:", private_key)

    # Encrypt plaintext
    plaintext = int(input("Please input an integer: "))
    ciphertext = elgamal_encrypt(public_key, plaintext)
    print("Ciphertext:", ciphertext)

    # Decrypt ciphertext
    decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
    print("Decrypted Text:", decrypted_text)
