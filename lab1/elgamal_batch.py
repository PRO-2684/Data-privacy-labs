from elgamal import elgamal_key_generation, elgamal_encrypt, elgamal_decrypt
from random import randint
from time import time_ns
from multiprocessing import Pool
from functools import partial

if __name__ == "__main__":

    MAX_PLAINTEXT = 2 ** 31 - 1
    SIZE = 100000

    # Set key_size, such as 256, 1024...
    key_size = 128

    # Generate keys
    public_key, private_key = elgamal_key_generation(key_size)
    print("Public Key:", public_key)
    print("Private Key:", private_key)
    plaintexts = [randint(0, MAX_PLAINTEXT) for _ in range(SIZE)]

    print("Using for loop:")
    t1 = time_ns()
    ciphertexts = []
    for plaintext in plaintexts:
        ciphertext = elgamal_encrypt(public_key, plaintext)
        ciphertexts.append(ciphertext)
    t2 = time_ns()
    decrypted_texts = []
    for ciphertext in ciphertexts:
        decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
        decrypted_texts.append(decrypted_text)
    t3 = time_ns()
    assert plaintexts == decrypted_texts, "Decryption error!"
    print(f"  Encryption time: {(t2 - t1) / 1000000} ms")
    print(f"  Decryption time: {(t3 - t2) / 1000000} ms")

    encrypt_func = partial(elgamal_encrypt, public_key)
    decrypt_func = partial(elgamal_decrypt, public_key, private_key)

    print("Using pool:")
    t1 = time_ns()
    pool = Pool()
    ciphertexts = pool.map(encrypt_func, plaintexts)
    pool.close()
    pool.join()
    t2 = time_ns()
    pool = Pool()
    decrypted_texts = pool.map(decrypt_func, ciphertexts)
    pool.close()
    pool.join()
    t3 = time_ns()
    assert plaintexts == decrypted_texts, "Decryption error!"
    print(f"  Encryption time: {(t2 - t1) / 1000000} ms")
    print(f"  Decryption time: {(t3 - t2) / 1000000} ms")
