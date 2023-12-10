# Write a bubble sort function
# that takes in a list and returns a sorted list.
# Use the test code below to check your function.
# print(bubble_sort([37, 42, 9, 19, 35, 4, 53, 22]))
# print(bubble_sort([5, 4, 3, 2, 1]))

def bubblesort(input_list):
    for i in range(len(input_list)):
        for j in range(len(input_list)-1):
            if input_list[j] > input_list[j+1]:
                input_list[j], input_list[j+1] = input_list[j+1], input_list[j]
    return input_list

print(bubblesort([37, 42, 9, 19, 35, 4, 53, 22]))
print(bubblesort([5, 4, 3, 2, 1]))
