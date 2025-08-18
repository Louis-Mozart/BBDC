def compute_accuracy(dat1,dat2,prof_skeleton):
    '''Compute the accuracy based on the test file: prof_skeleton'''
    acc_affect = 0
    acc_context = 0
    n_affect_true = 0
    n_context_true = 0

    for i in range(len(prof_skeleton)):
        if prof_skeleton["affect"][i] == True:

            n_affect_true += 1

            if dat1["affect"][i] == dat2["affect"][i]:

                acc_affect+=1


        if prof_skeleton["context"][i] == True:

            n_context_true += 1

            if dat1["context"][i] == dat2["context"][i]:

                acc_context+=1

    acc = (acc_affect/n_affect_true + acc_context/n_context_true)/2

    return acc

