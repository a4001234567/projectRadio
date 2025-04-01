import numpy as np
import scipy.stats as stats
from utils.funcPhi import func_phi,func_phi_inv


def apply_rho(msg_distribution,func_rho:dict):
    rho_result = 0
    for exp,coef in func_rho.items():
        rho_result += coef*func_phi_inv(1-np.pow(1-func_phi(msg_distribution),exp-1))
    return rho_result

def apply_lambda(msg_distribution,func_lambda:dict):
    lambda_result = 0
    for exp,coef in func_lambda.items():
        lambda_result += coef*((exp-1)*msg_distribution)
    return lambda_result

def density_evolution(sigmaS,func_rho,func_lambda):
    '''
    func_rho is the degree distribution of check nodes
    func_lambda is the degree distribution of varaible nodes
    '''
    initial_msg = 2/sigmaS
    var_distribution = initial_msg
    for l in range(5):
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        cdf_val = stats.norm(loc=var_distribution,scale=np.sqrt(2.*var_distribution)).cdf(0)
        if cdf_val < 1e-4 or (cdf_val > 0.1 and l >= 1):
            break
    return 1-cdf_val

def find_thres(func_rho,func_lambda):
    '''
    dc,dv
    '''
    threshold = 0
    for i in range(2,-10,-1):
        if density_evolution(threshold+pow(2,i),func_rho,func_lambda) >= 1 - 1e-4:
            threshold += pow(2,i)
    return threshold

# Derive the below two functions from the cooresponding equations:
# Eb/N0 = 1/(2 R sigma^2)
# Reference: Ryan, William, and Shu Lin. "Channel codes: classical and modern." Cambridge University press, 2009.
# Chapter 1:1.5.1.3:Page 15
def convert_snr_to_ebno(snr,R):
    return -10*np.log10(2*R*snr)

def convert_ebno_to_snr(ebno,R):
    return 2*R*np.power(10,-ebno/10)


if __name__ == '__main__':
    #test_sum_pow()
    #exit()
    np.seterr(invalid='raise')
    import time
    start = time.time()
    print(np.sqrt(find_thres({10:1.},{5:1.})))
    print(convert_snr_to_ebno(find_thres({10:1.},{5:1.}),0.5))
    print(np.sqrt(find_thres({6:1.},{4:1.})))
    print(convert_snr_to_ebno(find_thres({6:1.},{4:1.}),1/3))
    print(np.sqrt(find_thres({5:1.},{3:1.})))
    exit()
    #snr = np.sqrt(find_thres({7:0.234,6:0.766},{2:0.332,3:0.247,4:0.110,5:0.311}))
    snr = find_thres({2:0.239,3:0.295,4:0.033,11:0.433},{8:0.57,7:0.43})
    print(f'EbNo: {convert_snr_to_ebno(snr,0.5)}, SNR: {snr}')
    #snr = np.sqrt(find_thres({7:0.57,6:0.43},{1:0.239,2:0.295,3:0.033,10:0.433}))
    #print(f'EbNo: {convert_snr_to_ebno(snr,0.5)}, SNR: {snr}')
    #print(np.sqrt(find_thres({6:0.234,5:0.766},{1:0.332,2:0.247,3:0.110,4:0.311})))
    #snr = np.sqrt(find_thres({20:0.7,19:0.3},{8:0.1284,7:0.2907,3:0.3406,2:0.2343}))
    #print(f'EbNo: {convert_snr_to_ebno(snr,0.82)}, SNR: {snr}')
    print(time.time()-start)
    #dc,dv
