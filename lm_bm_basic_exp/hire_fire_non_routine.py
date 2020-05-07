from toolbox import *
from hire_fire_routine import delete_from_old_r_job
from sys import exit


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


def f_fires_nr_workers(h_arr, fired_ids, f):
    for h in h_arr[fired_ids]:
        h.fired = True
        f.n_nr_fired += 1


def firms_fire_nr_workers(v_mat, h_arr, f_arr, emp_mat, nr_job_arr):

    f_mask = v_mat[:, 2] < 0
    val = np.sum(f_mask)
    h_inds = np.arange(len(h_arr))
    if val > 0:
        fire_arr = v_mat[f_mask]
        ids = fire_arr[:, 0]
        n_fire_arr = fire_arr[:, 2] * (-1)
        for i in range(len(ids)):
            # get id of the firm, and number of workers it wants to fire
            f_id, n = int(ids[i]), int(n_fire_arr[i])
            emp_mask = emp_mat[f_id, :] > 0
            # get employees as object
            emp_ids = h_inds[np.logical_and(emp_mask, nr_job_arr)]
            # look at wages of the employee
            wages = np.array([h.w for h in h_arr[emp_ids]])
            # take indices of employees with highest wages
            mask = np.argsort(wages)[-n:] # indices in emp array
            fired_ids = emp_ids[mask]
            f_fires_nr_workers(h_arr, fired_ids, f_arr[f_id])




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
            # Keep application that are not in "chosen_apps"
            f.apps_nr = f.apps_nr[bool_arr]

        # delete non-routine workers only from routine application queues
        # if they got a routine job first -> use "not app_id in emp_ids"
        if len(f.apps_r) > 0:
            r_f_h_app_ids = f.apps_r[:, 0].astype(int)
            bool_arr = np.array([not app_id in emp_ids for app_id in r_f_h_app_ids])
            f.apps_r = f.apps_r[bool_arr]


def employ_nr_apps(h_arr, emp_mat, app_mat, nr_job_arr, f, lambda_LM, min_w, t):
    for h in h_arr:
        h.job_offer[t] = 1
        # delete all applications
        app_mat[:, h.id] = np.zeros(len(app_mat[:, h.id]))
        # either they already have a nr job
        if nr_job_arr[h.id]:
            Pr = Pr_LM(h.w, h.d_w, lambda_LM)
            switch = bool(draw_one(Pr))
            if switch:
                f.v_nr -= 1
                # delete from old nr job
                emp_mat[:, h.id] = np.zeros(len(emp_mat[:, h.id]))
                # household gets employed
                emp_mat[f.id, h.id] = 1
                h.w = np.maximum(h.d_w, min_w)
                # update_desired wage in case of a minimum wage
                h.d_w = h.w
                h.fired_time = 0
                h.fired = False

        # or they are unemployed or have a routine job
        else:
            f.v_nr -= 1
            boolean = np.sum(emp_mat[:, h.id]) > 0
            if boolean:
                # delete from old "r" job
                emp_mat[:, h.id] = np.zeros(len(emp_mat[:, h.id]))

            h.u[t] = 0
            # household gets employed
            emp_mat[f.id, h.id] = 1
            h.w = np.maximum(h.d_w, min_w)
            # update_desired wage in case of a minimum wage
            h.d_w = h.w
            nr_job_arr[h.id] = True


def delete_from_old_nr_job(h, emp_mat):
    emp_mat[:, h.id] = np.zeros(len(emp_mat[:, 0]))


def firms_employ_nr_applicants(m):
    f_arr, h_arr, lambda_LM, min_w, t = m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t
    emp_matrix, routine_arr, nr_job_arr = m.emp_matrix, m.routine_arr, m.nr_job_arr
    app_matrix = m.app_matrix
    # 1. get vacancies
    # v_arr = np.array([f.v for f in f_arr])

    # 2. get ids of demand side
    f_ids = np.arange(len(f_arr))

    # 3. shuffle ids
    rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)

    # 4. shuffle vacancies using shuffled ids
    rand_v_arr = np.array([f.v_nr for f in f_arr[rand_f_ids]])

    # get demanded wages
    d_wages = np.asarray([h.d_w for h in m.h_arr])
    h_ids = np.arange(len(h_arr))

    val = True
    while val:
        # print("in 'firms_employ_nr_applicants'")
        v_arr = np.array([f.v_nr if f.v_nr > 0 else 0 for f in f_arr])
        val_arr = v_arr @ app_matrix[:, m.non_routine_arr]
        # some_arr = np.array(
        #     [(f.id, f.v_nr, f.n_nr_fired, f.Nnr, f.d_Nnr) for f in m.f_arr]
        # )
        # print("val_arr: {}".format(some_arr))
        val = np.sum(v_arr @ app_matrix[:, m.non_routine_arr])

        for i in range(len(f_ids)):

            id, v = int(rand_f_ids[i]), int(rand_v_arr[i])
            if np.sum(app_matrix[id, m.non_routine_arr]) * (v > 0) > 0:
                # get app ids
                applied = app_matrix[id, :] > 0 # look at all applicants
                # take ids of workers that have applied and are non-routine type
                mask = np.logical_and(applied, m.non_routine_arr)
                h_app_ids = h_ids[mask]
                # sort app ids from lowest to highest wrt to wages
                sorted_app_ids = np.argsort(d_wages[h_app_ids])
                # choose cheapest ids
                sorted_h_ids = h_app_ids[sorted_app_ids] # sort applicant ids
                chosen_apps = sorted_h_ids[0:v]
                # set only column app_mat[:, h_id] to zero if worker h_id rejects
                employ_nr_apps(h_arr[chosen_apps], emp_matrix, app_matrix,
                               nr_job_arr, f_arr[id], lambda_LM, min_w, t)

        update_N(f_arr, emp_matrix, nr_job_arr)
        update_v(f_arr)
        rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)
        rand_v_arr = np.array([f.v_nr for f in f_arr[rand_f_ids]])