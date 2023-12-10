from elgamal import elgamal_key_generation, elgamal_encrypt, elgamal_decrypt
from time import time_ns

key_sizes = [64, 128, 256]
TEXT = 123456789
ROUND = 16

def test(key_size: int) -> (int, int, int):
    print("  Key size:", key_size)
    t1 = time_ns()
    public_key, private_key = elgamal_key_generation(key_size)
    t2 = time_ns()
    print(f"    Key generation time: {(t2 - t1) / 1000000} ms")
    ciphertext = elgamal_encrypt(public_key, TEXT)
    t3 = time_ns()
    print(f"    Encryption time: {(t3 - t2) / 1000000} ms")
    plaintext = elgamal_decrypt(public_key, private_key, ciphertext)
    assert plaintext == TEXT, "Decryption error!"
    t4 = time_ns()
    print(f"    Decryption time: {(t4 - t3) / 1000000} ms")
    return (t2 - t1, t3 - t2, t4 - t3)

def batch():
    result = {}
    for key_size in key_sizes:
        times = test(key_size)
        result[key_size] = times
    return result

if __name__ == "__main__":
    final = {key_size: [0, 0, 0] for key_size in key_sizes}
    for i in range(ROUND):
        print(f"Round {i}:")
        result = batch()
        for key_size in key_sizes:
            for i in range(3):
                final[key_size][i] += result[key_size][i]
    print("")
    print("Average:")
    for key_size in key_sizes:
        for i in range(3):
            final[key_size][i] /= ROUND
        print(f"  Key size: {key_size}")
        print(f"    Key generation time: {final[key_size][0] / 1000000} ms")
        print(f"    Encryption time: {final[key_size][1] / 1000000} ms")
        print(f"    Decryption time: {final[key_size][2] / 1000000} ms")
