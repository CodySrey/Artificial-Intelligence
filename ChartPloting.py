# -*- coding: utf-8 -*-
"""
Illustration of chart and graph ploting in python. These are multiple way to 
draw your chart/graph. Each example (Ex#) is self-contained, uncomment out one 
example at a time to test.

More API documentation here: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.html
    
@author: knguyen27
"""
import matplotlib.pyplot as plt


#giving labels to your graph
plt.xlabel('x - axis')
plt.ylabel('y - axis')

#giving a title to your graph
plt.title('Plotting Examples!')

#coordinates of your graph - should have the same number of values
x = [1, 2, 3, 4, 5, 6, 7, 8, 9] #x-axis list
y = [2, 4, 1, 5, 6, 7, 2, 3, 8] #y-axis list


"""Below are example of some plotting function - the input  
must match the requirement of each function as you see in the latter examples
"""

"""Ex1:  - draw the coordiante as a line """
#plt.plot(x,y)  #plot the x-y coordinates

""" EX2:  you can provide the third parameter for the graph label """
#plt.plot(x,y, label="line 1")

"""Ex3: plot the dots """
#plt.scatter(x,y) # 

""" Ex 4: you can provide color and symbol used and the symbol size for your points """
#plt.scatter(x,y, label= "points", color = "green",  marker ="*", s = 150)

"""Ex 5: draw bar chart """
#plt.bar(x,y)

"""Ex 6: put tick labels and color on the bars """
#labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'M']
#plt.bar(x, y, tick_label=labels, width=0.8, color=['red', 'green'])

"""Ex 7: draw bar sideway """
#labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'M']
#plt.barh(x, y, tick_label=labels,  color=['red', 'green'])

"""We need different kind of data for the following plots """

"""Ex 8: draw pie chart """
#proportions = [15, 30, 50, 5] #each slice of the pie
#colors = ['r', 'g', 'b', 'y'] #coresponding color of each slice
#itemNames = ['A', 'B', 'C', 'D']
#plt.pie(proportions, labels = itemNames, colors=colors, startangle=90, 
#        shadow = True, explode = (0, 0, 0.1, 0),radius = 1.2, autopct = '%1.1f%%')

"""Ex 9: histogram drawing, similar data are grouped into a bin -> drawn as a taller bar"""
#prices = [14, 3, 31, 6, 8, 2, 3, 7, 4, 1, 3, 57, 23, 3, 5, 2, 5, 11 , 45, 2, 43, 55]
#ranges = (1, 100)
#bins = 10
#plt.hist(prices, bins, ranges, color='blue', histtype='bar', rwidth=0.5)


plt.legend() #show the legend for the drawing
plt.show() #display the graph































