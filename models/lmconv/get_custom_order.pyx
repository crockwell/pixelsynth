import numpy as np 
import heapq

def custom_idx(rows, cols, distances, mass_center):
    """begin from maximum distance to background pixels
    and proceed towards background pixels; fill in these
    closest to foreground pixels, then furtherst
    ties are broken using spiral pattern, 
    which starts from center of mass. 
    One constraint is each new pixel must touch a pixel 
    previously predicted (L,R,U,D); so that the mask is not blank.
    
    inputs: rows, cols - int of sizes
    distances: (rows, cols) of distance to background
    (if in foreground -- positive), 
    to foreground (if in background -- negative)
    mass_center: row, col int tuple of start of spiral 
    """
    idx = []
    r = mass_center[0]
    c = mass_center[1]
    diff = c - r
    tot = mass_center[0] + mass_center[1]
    assert(rows == cols)

    distances *= 10000

    '''
    # get spiral ordering &
    # combine spiral order with distances
    # spiral order used only to break ties.
    # multiply distance by 10000
    order = 0
    while order < rows * cols:
        if r >= 0 and c >= 0 and r < rows and c < cols:
            distances[r,c] -= order
            #idx.append((r, c))
            order += 1
            #print(r,c, order, rows*cols)
        if c >= r + diff and c+r < tot: # right
            c += 1
        elif c > r + diff and c+r >= tot: # down
            r += 1
        elif c <= r + diff and c+r >= tot: # left
            c -= 1
        elif c < r + diff and c+r < tot: # up
            r -= 1
    '''

    # start at highest distance, add elements to list if allowed
    # allowed only if neighbor a current pixel
    #r = mass_center[0]
    #c = mass_center[1]

    c = np.argmax(distances) % rows
    r = int((np.argmax(distances)-c) / rows)
    final_order = [[r, c]]
    #final_distances = []
    used = [[r, c]]
    candidate_distances = []
    #import pdb 
    #pdb.set_trace()
    while len(final_order) < rows * cols:
        # add candidates surrounding new 
        if r - 1 >= 0 and [r-1,c] not in used: # Up
            heapq.heappush(candidate_distances,(-distances[r-1,c], [r-1,c])) 
            used.append([r-1,c])
            #candidate_distances.append(distances[r-1,c])
        if r + 1 < rows and [r+1,c] not in used: # Down
            heapq.heappush(candidate_distances,(-distances[r+1,c], [r+1,c])) 
            used.append([r+1,c])
            #candidate_distances.append(distances[r+1,c])
        if c - 1 >= 0 and [r,c-1] not in used: # Left 
            heapq.heappush(candidate_distances,(-distances[r,c-1], [r,c-1])) 
            used.append([r,c-1])
            #candidate_distances.append(distances[r,c-1])
        if c + 1 < cols and [r,c+1] not in used: # Right
            heapq.heappush(candidate_distances,(-distances[r,c+1], [r,c+1])) 
            used.append([r,c+1])   
            #candidate_distances.append(distances[r,c+1])
        (_, [r,c]) = heapq.heappop(candidate_distances)
        final_order.append([r, c])
        #final_distances.append(distances[r, c])
        #pdb.set_trace()
        '''
        for (r_, c_) in final_order:
            if r_ - 1 >= 0 and [r_-1,c_] not in final_order: # Up
                candidates.append([r_-1,c_])
                candidate_distances.append(distances[r_-1,c_])
            if r_ + 1 < rows and [r_+1,c_] not in final_order: # Down
                candidates.append([r_+1,c_])
                candidate_distances.append(distances[r_+1,c_])
            if c_ - 1 >= 0 and [r_,c_-1] not in final_order: # Left 
                candidates.append([r_,c_-1])
                candidate_distances.append(distances[r_,c_-1])
            if c_ + 1 < cols and [r_,c_+1] not in final_order: # Right
                candidates.append([r_,c_+1])   
                candidate_distances.append(distances[r_,c_+1])
        argmax = np.argmax(candidate_distances)
        (r, c) = candidates[argmax]
        final_order.append([r, c])
        '''
        
        #pdb.set_trace()
        #U = -999999 if r - 1 < 0 else distances[r - 1, c]
        #L = -999999 if c - 1 < 0 else distances[r, c - 1]
        #R = -999999 if c - 1 >= cols else distances[r, c + 1]
        #D = -999999 if r + 1 >= rows else distances[r + 1, c]
        '''
        argmax = np.argmax([U,L,R,D])
        if argmax == 0:
            r -= 1
        elif argmax == 1:
            c -= 1
        elif argmax == 2:
            c += 1
        elif argmax == 3:
            r += 1
        '''

    ### step 3: make new index of ordering
    #idx = np.dstack(np.unravel_index(np.argsort(distances.ravel())[::-1], (rows, cols)))[0]

    return np.array(final_order)