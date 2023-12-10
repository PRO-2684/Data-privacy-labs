from elgamal import elgamal_key_generation, elgamal_encrypt, elgamal_decrypt

key_size = 256
public_key, private_key = elgamal_key_generation(key_size)
print("Public key: ", public_key)
print("Private key: ", private_key)

# Test randomness of elgamal algorithm
plaintext = 1234
ciphertext1 = elgamal_encrypt(public_key, plaintext)
print("Ciphertext 1: ", ciphertext1)
ciphertext2 = elgamal_encrypt(public_key, plaintext)
print("Ciphertext 2: ", ciphertext2)
print("Ciphertext 1 == Ciphertext 2: ", ciphertext1 == ciphertext2)

# Test multiplicative homomorphism of elgamal algorithm
plaintext1 = 1234
plaintext2 = 5678
ciphertext1 = elgamal_encrypt(public_key, plaintext1)
print("Ciphertext 1: ", ciphertext1)
ciphertext2 = elgamal_encrypt(public_key, plaintext2)
print("Ciphertext 2: ", ciphertext2)
ciphertext3 = (
    ciphertext1[0] * ciphertext2[0] % public_key[0],
    ciphertext1[1] * ciphertext2[1] % public_key[0],
)
print("Ciphertext 3: ", ciphertext3)
plaintext3 = elgamal_decrypt(public_key, private_key, ciphertext3)
print("Plaintext 3: ", plaintext3)
mul = plaintext1 * plaintext2
print("Plaintext 1 * Plaintext 2: ", mul)
print("Plaintext 3 == Plaintext 1 * Plaintext 2: ", plaintext3 == mul)
