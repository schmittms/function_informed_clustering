import numpy as np
from sklearn.linear_model import LinearRegression


# This can readily be transformed into a function that only takes grouping as argument, 
# def loss_fct(g): return compute_regression_loss(g, abundances, target)[0]
def compute_regression_loss(grouping, abundances, target):
    # abundances shape: [n_samples, n_strains]
    # target shape: [n_samples, n_dof]
    # grouping shape: [n_strains,]
    n_groups = np.max(grouping).astype(int) + 1
    
    proj_mat = np.zeros((abundances.shape[1], n_groups))
    for i in range(n_groups):
        proj_mat[grouping==i, i] = 1.

    group_abundances = abundances @ proj_mat

    reg = LinearRegression()
    reg.fit(group_abundances, target)
    predictions = reg.predict(group_abundances)
    # Compute the regression loss
    err = np.mean((target - predictions)**2)

    R2 = 1 - (np.mean((target - predictions)**2) / np.mean((target - np.mean(target, axis=0))**2))
    return err, reg.coef_, reg.intercept_, R2, predictions


def metropolis(loss_fct, # function that we use to evaluate the grouping
               n_strains=48, n_steps=1000, n_groupings=20, max_n_groups=20, min_n_groups=2):
    """
    This algorithms works on a list of groupings, each of which is a list of integers:
        [1,1,1,0,0,0,1,0,0] # split into two groups ("valency"=2)
    G = [1,1,2,0,2,0,0,2,2] # split into three groups ("valency"=3)
        [3,1,2,1,0,2,0,3,2] # split into four groups ("valency"=4)

    Each grouping has an associated loss, 
        [0.65]
    E = [0.32]
        [0.19]
    
    At each step, we select one of these groupings and form a new grouping by either splitting or merging it.
    At step i we might select, for example
        selected_grouping = [1,1,2,0,2,0,0,2,2]

    Then if we perform the "split" operation, we randomly select either one of the cluster identities [0,1,2],
    for example 
        group_to_split = 1

    Then we randomly select half of the members of this group to assign to a new group, so that
        selected_grouping = [1,1,2,0,2,0,0,2,2] -> [3,1,2,0,2,0,0,2,2]

    If we perform the "merge" operation, we randomly select two of the cluster identities [0,1,2],
    for example 
        groups_to_merge = [0,2]

    Then we merge these two groups into one group, so that
        selected_grouping = [1,1,2,0,2,0,0,2,2] -> [1,1,0,0,0,0,0,0,0]

    Using the selected grouping, we compute the loss (up to you how to do this in particular).

    We compare this loss to the previous loss for the grouping of the same valency as the *new* grouping.
    For example in the split example above, we go from valency 3 to valency 4. We do not update the loss for the valency 3 grouping (E[1] = 0.32), but the loss for valency 4 (E[2] = 0.19), even though we started with a valency 3 grouping.
    
    """

    # Randomly initialize a list of groupings
    init_group_valencies = np.arange(n_groupings) + 2 # how many unique groups to start with

    groupings = np.zeros((n_groupings, n_strains)) 
    for i in range(len(init_group_valencies)):
        # Randomly assign group identities to chunks of the row
        boundaries = [0, *np.random.choice(np.arange(1,n_strains), init_group_valencies[i]-1, replace=False), n_strains]
        boundaries = np.sort(boundaries)
        
        for group_id in range(init_group_valencies[i]):
            groupings[i, boundaries[group_id]:boundaries[group_id+1]] = group_id

        # Randomly permute the group identities
        groupings[i] = np.random.permutation(groupings[i]).astype(int)
         
        assert len(np.unique(groupings[i])) == init_group_valencies[i], "Grouping does not have intended valency, check indexing of group boundaries"+str(i)


    groupings_init = np.copy(groupings)
    E_min = [np.inf,] * n_groupings # store the minimum regression loss for each grouping
    E_hist = []


    for i in range(n_steps):
        group_idx = int(np.random.randint(n_groupings))
        selected_grouping = np.copy(groupings[group_idx]) # select one group 
        n_groups = np.max(selected_grouping).astype(int) + 1

        # Split group
        if n_groups<=min_n_groups or (n_groups<max_n_groups and np.random.rand()>0.5): # if we are at bottom, we *must* split
            group_identities, group_sizes = np.unique(selected_grouping, return_counts=True)
            group_to_split = int(np.random.choice(group_identities[group_sizes>1])) # select a group to split among those with more than one member
            group_size = group_sizes[group_identities==group_to_split].squeeze()
            group_member_indices = np.where(selected_grouping==group_to_split)[0]
            assert len(group_member_indices) == group_size, "group size does not match number of members"

            # deterministically assign half of the group to a new group
            new_ids = np.zeros(group_size, dtype=bool)
            new_ids[:group_size//2] = True
            new_ids = np.random.permutation(new_ids) # shuffle

            # randomly assign half of the group to a new group
            #new_ids = np.random.rand(group_size) < 0.5
            # make sure at least one 0 and one 1
            #if np.sum(new_ids)==0: 
            #    new_ids[np.random.randint(group_size)] = True
            #if np.sum(new_ids)==group_size:
            #    new_ids[np.random.randint(group_size)] = False

            selected_grouping[group_member_indices[new_ids]] = n_groups

        # Merge groups
        else:
            group_identities, group_sizes = np.unique(selected_grouping, return_counts=True)
            groups_to_merge = np.random.choice(group_identities[group_sizes>=1], size=2, replace=False).astype(int) # select two groups to merge

            selected_grouping[selected_grouping==groups_to_merge[1]] = groups_to_merge[0]
            selected_grouping[selected_grouping>groups_to_merge[1]] -= 1
            
        # Compute loss
        E_new = loss_fct(selected_grouping)

        # Reassign grouping if loss is lower
        new_group_idx = len(np.unique(selected_grouping)) - 1

        if E_new < E_min[new_group_idx]: # accept new grouping
            E_min[new_group_idx] = E_new
            groupings[new_group_idx, :] = selected_grouping.astype(int)

            
            assert len(np.unique(groupings[new_group_idx])) == np.max(groupings[new_group_idx]) + 1, \
            "grouping not valid after operation"

        E_hist.append(np.copy(E_min))

    return groupings, E_hist, groupings_init

