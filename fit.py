




def compute_abc_r(u1,u2,u3,u4,lamda,len_F,len_M):
    #u1     1*M
    #u2     1*M
    #u3     1*M
    #u4     1*M
    #k      F*M
    #lamda  1*M
    a_r=(u3*lamda+u4)/u1
    b_r=u2/u1