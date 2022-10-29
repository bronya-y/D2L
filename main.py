# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    criterion = nn.BCELoss(size_average=False, reduce=False)
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)

    loss_input = torch.sigmoid(input)
    output = criterion(loss_input, target)
    print(input)
    print(output)
    print(target)
    print(loss_input)


    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
