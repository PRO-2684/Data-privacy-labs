from elgamal import elgamal_key_generation, elgamal_encrypt, elgamal_decrypt
from time import time_ns
from random import getrandbits

KEY_SIZE = 256
TEXT_SIZE = 128
ROUND = 128
PUBLIC_KEY, PRIVATE_KEY = elgamal_key_generation(KEY_SIZE)
print("Public key: ", PUBLIC_KEY)
print("Private key: ", PRIVATE_KEY)

# Test randomness of elgamal algorithm
plaintext = 1234
ciphertext1 = elgamal_encrypt(PUBLIC_KEY, plaintext)
print("Ciphertext 1: ", ciphertext1)
ciphertext2 = elgamal_encrypt(PUBLIC_KEY, plaintext)
print("Ciphertext 2: ", ciphertext2)
print("* Ciphertext 1 == Ciphertext 2: ", ciphertext1 == ciphertext2)

# Test multiplicative homomorphism of elgamal algorithm
plaintext1 = 1234
plaintext2 = 5678
ciphertext1 = elgamal_encrypt(PUBLIC_KEY, plaintext1)
print("Ciphertext 1: ", ciphertext1)
ciphertext2 = elgamal_encrypt(PUBLIC_KEY, plaintext2)
print("Ciphertext 2: ", ciphertext2)
ciphertextMul = (
    ciphertext1[0] * ciphertext2[0] % PUBLIC_KEY[0],
    ciphertext1[1] * ciphertext2[1] % PUBLIC_KEY[0],
)
print("Ciphertext mul: ", ciphertextMul)
plaintextMul = elgamal_decrypt(PUBLIC_KEY, PRIVATE_KEY, ciphertextMul)
print("Plaintext mul: ", plaintextMul)
mul = plaintext1 * plaintext2 % PUBLIC_KEY[0]
print("Plaintext 1 * Plaintext 2: ", mul)
print("* Plaintext mul == Plaintext 1 * Plaintext 2: ", plaintextMul == mul)

# Running time analysis for multiplicative homomorphism
sum1 = 0
sum2 = 0
for i in range(ROUND):
    print(f"Round {i+1}", end="\r")
    plaintext1 = getrandbits(TEXT_SIZE)
    plaintext2 = getrandbits(TEXT_SIZE)
    ciphertext1 = elgamal_encrypt(PUBLIC_KEY, plaintext1)
    ciphertext2 = elgamal_encrypt(PUBLIC_KEY, plaintext2)
    ciphertextMul = (
        ciphertext1[0] * ciphertext2[0] % PUBLIC_KEY[0],
        ciphertext1[1] * ciphertext2[1] % PUBLIC_KEY[0],
    )
    t1 = time_ns()
    plaintextMul = elgamal_decrypt(PUBLIC_KEY, PRIVATE_KEY, ciphertextMul)
    t2 = time_ns()
    plaintext1 = elgamal_decrypt(PUBLIC_KEY, PRIVATE_KEY, ciphertext1)
    plaintext2 = elgamal_decrypt(PUBLIC_KEY, PRIVATE_KEY, ciphertext2)
    mul = plaintext1 * plaintext2 % PUBLIC_KEY[0]
    t3 = time_ns()
    assert plaintextMul == mul
    sum1 += t2 - t1
    sum2 += t3 - t2
print("Sum time of decrypt([a]*[b]):", sum1, "ns")
print("Sum time of decrypt([a])*decrypt([b]):", sum2, "ns")
print("Average time of decrypt([a]*[b]):", sum1 / ROUND, "ns")
print("Average time of decrypt([a])*decrypt([b]):", sum2 / ROUND, "ns")
