from zephyr import VectorAdder


def main() -> None:
    left = [1.0, 2.0, 3.0, 4.0]
    right = [10.0, 20.0, 30.0, 40.0]
    result = VectorAdder().add(left, right)

    assert result == [11.0, 22.0, 33.0, 44.0]
    print(f"CUDA vector add: {left} + {right} = {result}")


if __name__ == "__main__":
    main()
