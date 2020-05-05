from toolbox import *
from hire_fire_routine import delete_from_old_r_job


##################################################################################
########### functions for the hiring, firing and application mechanism ###########
##################################################################################
################################# NON-ROUTINE ####################################
##################################################################################



################################# application#####################################
##################################################################################

def firms_sort_nr_applications(f_arr):
    for f in f_arr:
        f.apps_nr = f.apps_nr.reshape(int(len(f.apps_nr) / 2), 2)
        # firms sort for desired wages
        wages = f.apps_nr[:,1]
        sorted_ids = np.argsort(wages)
        f.apps_nr = f.apps_nr[sorted_ids]


def hs_send_nr_apps(f_arr, nr_h_arr, chi, H_nr, H_r, beta):
    """
    Non-routine households send applications to firms that want to hire
    workers for non-routine jobs. Only non-routine type households can
    apply for this kind of jobs.
    :param f_arr: List of all firm objects
    :param nr_h_arr: List of non-routine households
    :param chi: Size of observed subset of firms (between 0 and 1)
    :param H_nr: Number of non-routine households
    :param ids: Ids of firms that want to hire non routine workers
    """
    h_indices = np.array([h.id for h in nr_h_arr])
    f_ids = np.arange(len(f_arr))
    h_arr_shuffled = nr_h_arr[rd.choice(h_indices, H_nr, replace=False) - H_r]
    subN = get_N_sub(chi, len(f_ids))
    for h in h_arr_shuffled:
        # write job application
        application = (h.id, (beta**h.exp)*h.d_w)
        # oberve  random subset of firms
        f_arr_shuffled = f_arr[rd.choice(f_ids, subN, replace=False)]
        # send application to observed firms
        for f in f_arr_shuffled:
            f.apps_nr = np.insert(f.apps_nr, 0, application)
    firms_sort_nr_applications(f_arr[f_ids])




################################## firing ####################################
##############################################################################


def get_nr_fired_ids(f, emp_ids):
    return f.nr_employees[emp_ids].astype(int)


def f_fires_nr_workers(h_arr, fired_ids, emp_ids, f, t):
    """
    Firm f fires routine type workers
    :param h_arr: List of all household objects
    :param fired_ids: List of the ids that firm f wants to fire
    :param emp_ids: List of the indices of the fired workers in
    firm f's "employee array"
    :param f: firm f as firm object
    :param t: current period number
    """
    # f.nr_employees = np.delete(f.nr_employees, emp_ids)
    # f.fired_nr_employees = fired_ids
    for h in h_arr[fired_ids]:
        h.fired = True
        h.fired_time_max = 2
        f.n_nr_fired += 1


def firms_fire_nr_workers(v_mat, h_arr, f_arr, t):
    """
    Firms fire non-routine workers.
    :param v_mat: Fx3 matrix (F is the number of firms) which
    includes the firm ids (1. column), number of workers to
    be hired (if positive) or fired (if negative) w.r.t.
    routine (2. column) resp. non-routine workers (3.) column.
    :param H_arr: List of household objects.
    :param F_arr: List of firm objects.
    :param t: Current period number
    """
    fire_arr = v_mat[v_mat[:, 2] < 0]
    ids = fire_arr[:, 0]
    n_fire_arr = fire_arr[:, 2] * (-1)
    for i in range(len(ids)):
        # get id of the firm, and number of workers it wants to fire
        f_id, n = int(ids[i]), int(n_fire_arr[i])
        # get employees as object
        emps = h_arr[f_arr[f_id].nr_employees.astype(int)]
        # look at wages of the employee
        wages = np.array([h.w for h in emps])
        # take indices of employees with highest wages
        emp_ids = np.argsort(wages)[-n:]
        fired_ids = get_nr_fired_ids(f_arr[f_id], emp_ids)
        f_fires_nr_workers(h_arr, fired_ids, emp_ids, f_arr[f_id], t)




################################## hiring ####################################
##############################################################################


def remove_nr_apps_from_queues(f_arr, chosen_apps, emp_ids):
    """
    Removes non-routine job applicants that were chosen by a firm for employment
    from other firms' application queues. Note that in this case we also remove from r queues.
    :param f_arr: List of firms that want to hire
    :param chosen_apps: List of household ids chosen for employment
    """
    for f in f_arr:

        if len(f.apps_nr) > 0:
            nr_f_h_app_ids = f.apps_nr[:, 0].astype(int)
            bool_arr = np.array([not app_id in chosen_apps for app_id in nr_f_h_app_ids])
            f.apps_nr = f.apps_nr[bool_arr]

        if len(f.apps_r) > 0:
            r_f_h_app_ids = f.apps_r[:, 0].astype(int)
            bool_arr = np.array([not app_id in emp_ids for app_id in r_f_h_app_ids])
            f.apps_r = f.apps_r[bool_arr]


def employ_nr_apps(h_arr, f_arr, f, lambda_LM, min_w, t):
    """
    Firm f employs non-routine job applicants.
    :param h_arr: List of household objects
    :param f: Firm object
    :param lambda_LM: Sensitivity parameter for switch probability.
    :param nr_vacancies: Numbers of vacancies per firm for non-routine jobs. (array)
    :param v_id: Index number of firm f's vacancy number in nr_vacancies.
    :param t: Current period number.
    :return: Updated numbers of vacancies for non-routine jobs (nr_vacancies).
    """
    emp_ids = np.array([])
    for h in h_arr:
        h.job_offer[t] = 1
        # either they already have a nr job
        if h.nr_job:
            Pr = Pr_LM(h.w, h.d_w, lambda_LM)
            switch = bool(draw_one(Pr))
            if switch:
                f.v_nr -= 1
                delete_from_old_nr_job(h, f_arr)
                h.employer_id = f.id
                h.w = np.maximum(h.d_w, min_w)
                if h.w > h.d_w:
                    h.d_w = h.w
                f.nr_employees = np.append(f.nr_employees, h.id)
                emp_ids = np.append(emp_ids, h.id)
        # or they are unemployed or have a routine job
        else:
            f.v_nr -= 1
            boolean = h.employer_id == None
            if not boolean:
                delete_from_old_r_job(h, f_arr)
            h.employer_id = f.id
            h.u[t] = 0
            h.w = np.maximum(h.d_w, min_w)
            if h.w > h.d_w:
                h.d_w = h.w
            h.nr_job = True
            f.nr_employees = np.append(f.nr_employees, h.id)
            emp_ids = np.append(emp_ids, h.id)
    return emp_ids


def delete_from_old_nr_job(h, f_arr):
    f_id = h.employer_id # get id of old employer
    emp_arr = f_arr[f_id].nr_employees
    h_i = np.where(emp_arr == h.id)[0][0] # get index in emp_arr
    f_arr[f_id].nr_employees = np.delete(emp_arr, h_i)


def firms_employ_nr_applicants(m):
    f_arr, h_arr, lambda_LM, min_w, t = m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t
    emp_matrix, routine_arr, nr_job_arr = m.emp_matrix, m.routine_arr, m.nr_job_arr
    # 1. get vacancies
    # v_arr = np.array([f.v for f in f_arr])

    # 2. get ids of demand side
    f_ids = np.arange(len(f_arr))

    # 3. shuffle ids
    rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)

    # 4. shuffle vacancies using shuffled ids
    rand_v_arr = np.array([f.v_nr for f in f_arr[rand_f_ids]])

    # 5. extract positive vacancy numbers
    bool_arr = rand_v_arr > 0
    # demand_arr = rand_v_arr[bool_arr]

    # get total number of applications
    tot_applicants = np.sum(np.array([len(f.apps_nr) for f in f_arr]))

    while (tot_applicants > 0) and (np.sum(bool_arr) > 0):
        # firms choose applicants
        for i in range(len(f_ids)):

            id, v = int(rand_f_ids[i]), int(rand_v_arr[i])
            if len(f_arr[id].apps_nr) * (v > 0) > 0:
                h_app_ids = f_arr[id].apps_nr[:, 0].astype(int)
                chosen_apps = h_app_ids[0:v]
                emp_ids = employ_nr_apps(h_arr[chosen_apps], f_arr, f_arr[id],
                                         lambda_LM, min_w, t)
                remove_nr_apps_from_queues(f_arr[rand_f_ids], chosen_apps, emp_ids)

        update_N(f_arr, emp_matrix, nr_job_arr)
        update_v(f_arr)
        rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)
        rand_v_arr = np.array([f.v_nr for f in f_arr[rand_f_ids]])
        bool_arr = rand_v_arr


        tot_applicants = np.sum(np.array([len(f.apps_nr) for f in f_arr[rand_f_ids[rand_v_arr > 0]]]))