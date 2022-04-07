import math
import numpy as np
from data_loader import get_data
from sklearn.metrics import r2_score

###
# Begin Derived Data function generators
###

def derived_lambda_r(VSW, Ne, m_p, Te, Ti, Vth_perp_ion, Bmag_avg, sigma_0, c, p1=1, p2=1, p3=1, C_0=1):
    #def derived_lambda_r(L_perp, d_e, rho_s, a=1, b=1, c=1, C_0=1):
    L_perp = derived_L_perp(VSW)
    d_e = derived_d_e(derived_Omega_pe(Ne, m_p, sigma_0), c)
    rho_s = derived_rho_s(derived_rho_i(Vth_perp_ion, derived_Omega_i(Bmag_avg, m_p)), Te, Ti)
    #tmp1 = p1*C_0*L_perp**(1/9)
    tmp1 = C_0 * L_perp ** (1 / 9 + p1)
    #tmp2 = ((p2*d_e)*(p3*rho_s))**(4/5)
    tmp2 = ((d_e ** p2) * (rho_s ** p3)) ** (4 / 5)

    return C_0*tmp1*tmp2

def derived_lambda_r_model(VSW, Ne, m_p, Te, Ti, Vth_perp_ion, Bmag_avg, sigma_0, c, beta, alpha=1, p1=1, p2=1, p3=1, C_0=1):
    #def lambda_r_model(rho_s, L_perp, d_e, beta, alpha=1, C=1):
    L_perp = derived_L_perp(VSW)
    d_e = derived_d_e(derived_Omega_pe(Ne, m_p, sigma_0), c)
    rho_s = derived_rho_s(derived_rho_i(Vth_perp_ion, derived_Omega_i(Bmag_avg, m_p)), Te, Ti)
    tmp1 = p1*C_0*rho_s
    #tmp2 = p2*(L_perp/rho_s)**alpha
    tmp2 = (L_perp / rho_s) ** (alpha + p2)
    #tmp3 = p3*(d_e/rho_s)**beta
    tmp3 = (d_e / rho_s) ** (beta + p3)

    return tmp1*tmp2*tmp3

def derived_VA(Bmag_avg, mu_0, m_p, Ni_avg):
    return (Bmag_avg**2)/((mu_0*m_p*Ni_avg)**(1/2))

def derived_Beta():
    return

def derived_Beta_para():
    return

def derived_Beta_perp():
    return

def derived_Omega_i(Bmag_avg, m_p):
    return (math.e*Bmag_avg)/m_p

def derived_Omega_pi():
    return

def derived_Omega_pe(Ne, m_p, sigma_0):
    return ((Ne*math.e)/(sigma_0*m_p))**(1/2)

def derived_rho_i(Vth_perp_ion, Omega_i):
    return Vth_perp_ion/Omega_i

def derived_rho_s(rho_i, Te, Ti):
    return rho_i * (Te/(2*Ti))**(1/2)

def derived_rho_c():
    return

def derived_d_i():
    return

def derived_d_e(omega_pe, c):
    return c/omega_pe

def derived_sigma_i():
    return

def derived_L_perp(VSW):
    return VSW/(2*math.pi*(10**-4))

def func(mu, sigma, factor, base_p1, base_p2, base_p3):
    results = []
    for i in range(0, 100):
        p1 = base_p1 + np.random.normal(mu, sigma) / factor
        p2 = base_p2 + np.random.normal(mu, sigma) / factor
        p3 = base_p3 + np.random.normal(mu, sigma) / factor

        lambda_r = derived_lambda_r(
            VSW=X['VSW'],
            Ne=X['Ne'],
            Te=X['Te'],
            Ti=X['Ti'],
            Vth_perp_ion=X['Vth_perp_Ion'],
            Bmag_avg=X['Bmag_avg'],
            sigma_0=constants['e'],
            m_p=constants['m_p'],
            c=constants['c'],
            p1=p1,
            p2=p2,
            p3=p3
        ).to_numpy()

        lambda_r_model = derived_lambda_r_model(
            VSW=X['VSW'],
            Ne=X['Ne'],
            Te=X['Te'],
            Ti=X['Ti'],
            Vth_perp_ion=X['Vth_perp_Ion'],
            Bmag_avg=X['Bmag_avg'],
            sigma_0=constants['e'],
            m_p=constants['m_p'],
            c=constants['c'],
            beta=derived_x['Beta'],
            p1=p1,
            p2=p2,
            p3=p3
        ).to_numpy()

        lambda_brk = y
        # r2 = r2_score(y_true=lambda_r, y_pred=y)
        r21 = r2_score(y_true=lambda_brk, y_pred=lambda_r)
        r22 = r2_score(y_true=lambda_r_model, y_pred=lambda_brk)
        r23 = r2_score(y_true=lambda_r_model, y_pred=lambda_r)
        #print("R21 = {0}, p1 = {1}, p2 = {2}, p3 = {3}".format(r21, p1, p2, p3))
        #print("R22 = {0}, p1 = {1}, p2 = {2}, p3 = {3}".format(r22, p1, p2, p3))
        #print("R23 = {0}, p1 = {1}, p2 = {2}, p3 = {3}".format(r23, p1, p2, p3))
        #print()

        results.append((r21, p1, p2, p3))

    argmax = np.argmax(np.array(results)[:, 0])
    return results[argmax]


if __name__ == '__main__':
    timestamps, X, derived_x, y, constants = get_data(split=False, return_constants=True)
    mu, sigma = 1, 1
    factor = 10
    base_p1 = 1
    base_p2 = 1
    base_p3 = 1

    for i in range (0,3):
        results = func(mu, sigma, factor, base_p1=base_p1, base_p2=base_p2, base_p3=base_p3)
        print(results)
        base_p1 = results[1]
        base_p2 = results[2]
        base_p3 = results[3]

    exit(0)
    for i in range(0,3):
        p1 = 1 + np.random.normal(mu, sigma) / factor
        p2 = 1 + np.random.normal(mu, sigma) / factor
        p3 = 1 + np.random.normal(mu, sigma) / factor


        lambda_r = derived_lambda_r(
            VSW=X['VSW'],
            Ne=X['Ne'],
            Te=X['Te'],
            Ti=X['Ti'],
            Vth_perp_ion=X['Vth_perp_Ion'],
            Bmag_avg=X['Bmag_avg'],
            sigma_0=constants['e'],
            m_p=constants['m_p'],
            c=constants['c'],
            p1=p1,
            p2=p2,
            p3=p3
            ).to_numpy()

        lambda_r_model = derived_lambda_r_model(
            VSW=X['VSW'],
            Ne=X['Ne'],
            Te=X['Te'],
            Ti=X['Ti'],
            Vth_perp_ion=X['Vth_perp_Ion'],
            Bmag_avg=X['Bmag_avg'],
            sigma_0=constants['e'],
            m_p=constants['m_p'],
            c=constants['c'],
            beta=derived_x['Beta'],
            p1=p1,
            p2=p2,
            p3=p3
            ).to_numpy()

        lambda_brk = y
        #r2 = r2_score(y_true=lambda_r, y_pred=y)
        r21 = r2_score(y_true=lambda_brk, y_pred=lambda_r)
        r22 = r2_score(y_true=lambda_r_model, y_pred=lambda_brk)
        r23 = r2_score(y_true=lambda_r_model, y_pred=lambda_r)
        print("R21 = {0}, p1 = {1}, p2 = {2}, p3 = {3}".format(r21, p1, p2, p3))
        print("R22 = {0}, p1 = {1}, p2 = {2}, p3 = {3}".format(r22, p1, p2, p3))
        print("R23 = {0}, p1 = {1}, p2 = {2}, p3 = {3}".format(r23, p1, p2, p3))
        print()

    mu, sigma = 0, 0.2
    rand_i = np.random.normal(mu, sigma, 100)
    models = []
    for val in rand_i:
        model = derived_lambda_r_model(
        VSW=X['VSW'],
        Ne=X['Ne'],
        Te=X['Te'],
        Ti=X['Ti'],
        Vth_perp_ion=X['Vth_perp_Ion'],
        Bmag_avg=X['Bmag_avg'],
        sigma_0=constants['e'],
        m_p=constants['m_p'],
        c=constants['c'],
        beta=derived_x['Beta'],
        alpha=val
        ).to_numpy()

        models.append(model)

    r2_scores = []
    for i, model in enumerate(models):
        r2_scores.append(r2_score(y_true=lambda_r_model, y_pred=lambda_r))

    argmax = np.argmax(r2_scores)
    print("Best lambda r,model: alpha = {0}, r2 = {1}".format(rand_i[argmax], r2_scores[argmax]))



