def sort_integers(nums):
    """Sort a list of integers in ascending order."""
    return sorted(nums)


# Test cases
if __name__ == "__main__":
    print(sort_integers([5, 2, 8, 1, 9]))  # [1, 2, 5, 8, 9]
    print(sort_integers([3, 3, 3, 3]))      # [3, 3, 3, 3]
    print(sort_integers([]))                # []
    print(sort_integers([1]))               # [1]
    print(sort_integers([9, 7, 5, 3, 1]))   # [1, 3, 5, 7, 9]