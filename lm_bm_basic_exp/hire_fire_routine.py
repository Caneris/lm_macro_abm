from toolbox import *


##################################################################################
########### functions for the hiring, firing and application mechanism ###########
##################################################################################
################################### ROUTINE ######################################
##################################################################################



################################ application #####################################
##################################################################################

def firms_sort_r_applications(f_arr):
    for f in f_arr:
        f.apps_r = f.apps_r.reshape(int(len(f.apps_r) / 2), 2)
        # firms sort for desired wages
        wages = f.apps_r[:,1]
        sorted_ids = np.argsort(wages)
        f.apps_r = f.apps_r[sorted_ids]


def hs_send_r_apps(f_arr, h_arr, chi, H, beta):
    """
    All households send applications to firms that want to hire
    workers for routine jobs. Both, routine and non-routine type
    households can apply for this kind of jobs.
    :param f_arr: List of all firm objects
    :param h_arr: List of all households
    :param chi: Size of observed subset of firms (between 0 and 1)
    :param H: Number of households
    :param ids: Ids of firms that want to hire routine workers
    """
    h_indices = np.array([h.id for h in h_arr])
    f_ids = np.arange(len(f_arr))
    h_arr_shuffled = h_arr[rd.choice(h_indices, H, replace=False)]
    subN = get_N_sub(chi, len(f_ids))
    for h in h_arr_shuffled:
        boolean = (not h.routine) and h.nr_job # non routine worker who has a non routine job
        if not boolean: # only apply if not a nr worker with an nr job
            # write job application
            application = (h.id, (beta**h.exp)*h.d_w)
            # oberve  random subset of firms
            f_arr_shuffled = f_arr[rd.choice(f_ids, subN, replace=False)]
            # send application to observed firms
            for f in f_arr_shuffled:
                f.apps_r = np.insert(f.apps_r, 0, application)
    firms_sort_r_applications(f_arr[f_ids])




################################## firing ####################################
##############################################################################


def get_r_fired_ids(f, emp_ids):
    return f.r_employees[emp_ids].astype(int)


def f_fires_r_workers(h_arr, fired_ids, f):
    for h in h_arr[fired_ids]:
        h.fired = True
        h.fired_time_max = 2
        f.n_r_fired += 1


def firms_fire_r_workers(v_mat, h_arr, f_arr, emp_mat, nr_job_arr):

    f_mask = v_mat[:, 1] < 0
    val = np.sum(f_mask)
    h_inds = np.arange(len(h_arr))
    if val > 0:
        fire_arr = v_mat[f_mask]
        ids = fire_arr[:, 0]
        n_fire_arr = fire_arr[:, 1] * (-1)
        for i in range(len(ids)):
            # get id of the firm, and number of workers it wants to fire
            f_id, n = int(ids[i]), int(n_fire_arr[i])
            emp_mask = emp_mat[f_id, :] > 0
            # get employees as object
            emp_ids = h_inds[np.logical_and(emp_mask, np.invert(nr_job_arr))]
            # look at wages of the employee
            wages = np.array([h.w for h in h_arr[emp_ids]])
            # take indices of employees with highest wages
            mask = np.argsort(wages)[-n:] # indices in emp array
            fired_ids = emp_ids[mask]
            f_fires_r_workers(h_arr, fired_ids, f_arr[f_id])




################################## hiring ####################################
##############################################################################


def remove_r_apps_from_queues(f_arr, chosen_apps):
    for f in f_arr:
        f_h_app_ids = f.apps_r[:, 0].astype(int)
        if len(f_h_app_ids) > 0:
            bool_arr = np.array([not app_id in chosen_apps for app_id in f_h_app_ids])
            f.apps_r = f.apps_r[bool_arr]


def employ_r_apps(h_arr, emp_mat, nr_job_arr, f, lambda_LM, min_w, t):

    for h in h_arr:
        h.job_offer[t] = 1
        if h.u[t] == 0:
            Pr = Pr_LM(h.w, h.d_w, lambda_LM)
            switch = bool(draw_one(Pr))
            if switch:
                f.v_r -= 1
                # delete from old r job
                emp_mat[:, h.id] = np.zeros(len(emp_mat[:, h.id]))
                # household gets employed
                emp_mat[f.id, h.id] = 1
                h.w = np.maximum(h.d_w, min_w)
                # update_desired wage in case of a minimum wage
                h.d_w = h.w
                nr_job_arr[h.id] = False
        else:
            f.v_r -= 1
            h.u[t] = 0
            # household gets employed
            emp_mat[f.id, h.id] = 1
            h.w = np.maximum(h.d_w, min_w)
            # update_desired wage in case of a minimum wage
            h.d_w = h.w


def delete_from_old_r_job(h, f_arr):
    f_id = h.employer_id # get id of old employer
    emp_arr = f_arr[f_id].r_employees
    h_i = np.where(emp_arr == h.id)[0][0] # get index in emp_arr
    f_arr[f_id].r_employees = np.delete(emp_arr, h_i)


def firms_employ_r_applicants(m):
    f_arr, h_arr, lambda_LM, min_w, t = m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t
    emp_matrix, routine_arr, nr_job_arr = m.emp_matrix, m.routine_arr, m.nr_job_arr
    # 1. get vacancies
    # v_arr = np.array([f.v for f in f_arr])

    # 2. get ids of demand side
    f_ids = np.arange(len(f_arr))

    # 3. shuffle ids
    rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)

    # 4. shuffle vacancies using shuffled ids
    rand_v_arr = np.array([f.v_r for f in f_arr[rand_f_ids]])

    # 5. extract positive vacancy numbers
    bool_arr = rand_v_arr > 0
    # demand_arr = rand_v_arr[bool_arr]

    # get total number of applications
    tot_applicants = np.sum(np.array([len(f.apps_r) for f in f_arr]))

    while (tot_applicants > 0) and (np.sum(bool_arr) > 0):
        # firms choose applicants
        for i in range(len(f_ids)):

            id, v = int(rand_f_ids[i]), int(rand_v_arr[i])
            if len(f_arr[id].apps_r) * (v > 0) > 0:
                h_app_ids = f_arr[id].apps_r[:, 0].astype(int)
                chosen_apps = h_app_ids[0:v]
                remove_r_apps_from_queues(f_arr[rand_f_ids], chosen_apps)
                employ_r_apps(h_arr[chosen_apps], emp_matrix, nr_job_arr,
                              f_arr[id], lambda_LM, min_w, t)

        update_N(f_arr, emp_matrix, nr_job_arr)
        update_v(f_arr)
        rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)
        rand_v_arr = np.array([f.v_r for f in f_arr[rand_f_ids]])
        bool_arr = rand_v_arr


        tot_applicants = np.sum(np.array([len(f.apps_r) for f in f_arr[rand_f_ids[rand_v_arr > 0]]]))
