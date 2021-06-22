import pickle
import os
import time


def big_results_writer(out_path, items, func_by_item, dry_run=False, verbose=False):
    # Write intermediate results to file in case there is a crash. Load the intermediate results if they exist.
    results_by_item = {}
    
    if os.path.exists(out_path):
        with open(out_path, 'rb') as fp:
            results_by_item = pickle.load(fp)

    count = 0
    start = time.time()
    for i, item in enumerate(items):
        if item not in results_by_item:
            result = func_by_item(item)

            if result is None:
                results_by_item[item] = None
            else:
                results_by_item[item] = result

            # Save intermediate data and print some results to show progress
            count += 1
            if count > 100:
                with open(out_path, 'wb') as fp:
                    pickle.dump(results_by_item, fp)
                count = 0
                now = time.time()
                if verbose:
                    print('Completed: {} {}  {:.2f} seconds'.format(i, item, now - start))
                start = now

            if dry_run and i > 200:
                break

        with open(out_path, 'wb') as fp:
            pickle.dump(results_by_item, fp)
