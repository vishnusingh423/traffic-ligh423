import unittest
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
def print_fail():
    printmd('**<span style="color: red;">TEST FAILED</span>**')    
def print_pass():
    printmd('**<span style="color: green;">TEST PASSED</span>**')
class Tests(unittest.TestCase):
    def test_one_hot(self, one_hot_function):
        try:
            self.assertEqual([1,0,0], one_hot_function('red'))
            self.assertEqual([0,1,0], one_hot_function('yellow'))
            self.assertEqual([0,0,1], one_hot_function('green'))
            except self.failureException as e:
            print_fail()
            print("Your function did not return the expected one-hot label.")
            print('\n'+str(e))
            return
        print_pass()
    def test_red_as_green(self, misclassified_images):
        for im, predicted_label, true_label in misclassified_images:
            if(true_label == [1,0,0]):
                
                try:
                    self.assertNotEqual(true_label, [0, 0, 1])
                except self.failureException as e:
                  
                    print_fail()
                    print("Warning: A red light is classified as green.")
                    print('\n'+str(e))
                    return

        print_pass()








