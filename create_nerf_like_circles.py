'''
I had some people ask for pseudocode to the make short looping circular videos seen in some recent view synthesis work. 
I've attached some below - hope someone finds it useful!
'''

# input and output are b,4,4 rotation matrices
new_output_RT = torch.zeros_like(input_RT)
# set output to same as input except 4th column which is 0,0,0,1 for now
# remember column 1-3 is rotation, 4th column is position
output_RT[:,:,:3] = input_RT[:,:,:3]
output_RT[:,3,3] = 1
# now, we set the position to the original position plus some scale times cyclical functions for x (horizontal), y (vertical), z (depth) 
# scale down z for this case since model is better at handling horizontal or vertical movement
output_RT[0,:3,3] = input_RT[0,:3,3] + .35 * torch.tensor([np.sin(2*np.pi*num/denom),np.cos(2*np.pi*num/denom),.4*np.sin(2*np.pi*(.25+num/denom))])
