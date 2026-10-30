import argparse
import pathlib
import time
import os

"""Evaluate kNN index files against a ground-truth.

This script reads k-nearest-neighbor index files and compares them with
ground-truth neighbor lists. The format is selected independently for the
index (`-I`) and ground truth (`-G`). A format consists of a base letter and
an optional `I` suffix:

    N   one line per point: neighbor IDs
    B   blocked two-line format: all neighbor-ID lines, then all distances
  M   two lines per point: neighbor IDs, then an auxiliary line (distances
      when `-D` is used)
  I   the first token on each neighbor-ID line is the source point ID

For example, an `N` file without `I` has one implicit source ID per line.
The first line below describes point 0 and the second describes point 1:

    7 13 5
    4 9 2

An `NI` file stores the source ID explicitly, so the lines may appear in any
source-ID order:

    42 7 13 5
    7 4 9 2

`B` is a blocked two-line format: all neighbor-ID lines come first, followed
by all distance lines. `B` uses implicit source IDs, while `BI` puts the source
ID at the start of each neighbor-ID line:

    # B: sources 0 and 1, then their distances
    7 13 5
    4 9 2
    0.12 0.31 0.48
    0.20 0.45 0.81

    # BI: source IDs are explicit
    42 7 13 5
    7 4 9 2
    0.00 0.12 0.31 0.48
    0.00 0.20 0.45 0.81

`M` is always read as pairs of lines, with each neighbor-ID line followed by
one auxiliary line. For example, an `MI` file is:

    42 7 13 5
    0.00 0.12 0.31 0.48
    7 4 9 2
    0.00 0.20 0.45 0.81

For `BI` and `MI`, the first token on the distance/auxiliary line is a dummy
value corresponding to the source ID; it is ignored. The remaining values
correspond positionally to the neighbor IDs. For `B` and `M` without `I`, the
distance line contains only neighbor distances. Use `-D` only when both inputs
provide at least `k` distances per point;
the IDs are then parsed for validation, but correctness is based on whether
each index distance is less than or equal to the ground-truth k-th distance.

If `-I` or `-G` is omitted, the defaults are `NI` for the index and `N` for
the ground truth. Implicit IDs start at 0. When a
directory is supplied, files are processed in sorted filename order; multiple
files therefore require explicit IDs (`I`) for unambiguous matching.

Usage examples:

  # Default: index has explicit IDs; ground truth has implicit IDs.
  python tools/evalulate_knng.py -i index.txt -g groundtruth.txt -k 10

  # Explicit IDs in several files.
  python tools/evalulate_knng.py -i index_dir -g gt_dir -k 20 -I NI -G NI

    # Compare distances from blocked files instead of neighbor IDs.
    python tools/evalulate_knng.py -i index.b -g groundtruth.b -k 10 -I B -G B -D
"""

FORMAT_BASES = {'B', 'M', 'N'}


def parse_args():
    """Parse command-line arguments.

    Returns a namespace with parsed arguments. See the module-level
    docstring for usage examples and format explanations.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate kNN index')

    parser.add_argument('-i', '--index',
                        dest='index_file_path',
                        required=True, action='store',
                        help='Path to a kNN index file or a directory containing the kNN index files')
    parser.add_argument('-g', '--ground-truth',
                        dest='ground_truth_file_path',
                        required=True, action='store',
                        help='Path to the ground truth index file')
    # k: the number of nearest neighbors to be evaluated
    parser.add_argument('-k', '--k',
                        dest='k',
                        required=True, action='store', type=int,
                        help='The number of nearest neighbors to be evaluated')

    parser.add_argument('-I', '--index-format',
                        dest='index_format',
                        required=False, action='store',
                        help='Format of the kNN index files: B/M/N plus optional I (default: NI)')
    parser.add_argument('-G', '--ground-truth-format',
                        dest='ground_truth_format',
                        required=False, action='store',
                        help='Format of the ground truth files: B/M/N plus optional I (default: N)')

    parser.add_argument('-D', '--dist-eval',
                        dest='use_distance_eval',
                        required=False, action='store_true',
                        help='Enable distance-based evaluation: a neighbor is correct if its distance is <= ground-truth k-th neighbor distance')

    parser.add_argument('--sparse-ground-truth',
                        dest='sparse_ground_truth',
                        required=False, action='store_true',
                        help='Enable sparse ground truth handling: missing points in the ground truth will be skipped instead of raising an error')

    args = parser.parse_args()
    if args.index_format is None:
        args.index_format = 'NI'
    if args.ground_truth_format is None:
        args.ground_truth_format = 'N'
    try:
        args.index_format = parse_format(args.index_format)
        args.ground_truth_format = parse_format(args.ground_truth_format)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def find_files(path):
    """Return a list of pathlib.Path objects for a file or directory.

    If `path` is a file path, return a single-element list. If it's a
    directory, return a sorted list of entries inside it.
    """
    if not os.path.isdir(path):
        return [pathlib.Path(path)]
    return sorted(pathlib.Path(path) / f for f in os.listdir(path))


def parse_format(fmt):
    """Parse a short format string into a dict.

    The format string must include one of the base letters `B`, `M`, or
    `N`. Optionally include `I` to indicate per-line IDs (e.g. "NI").
    Returns a dict with keys: `base`, `has_id`, and `spec`.
    """
    if fmt is None:
        raise ValueError('Format is missing')
    s = fmt.strip().upper()
    if not s:
        raise ValueError('Format is empty')
    has_id = 'I' in s
    base = None
    for ch in s:
        if ch == 'I':
            continue
        if ch in FORMAT_BASES:
            if base is not None and base != ch:
                raise ValueError(f'Format has multiple base letters: {fmt}')
            base = ch
        else:
            raise ValueError(f'Invalid format letter: {ch}')
    if base is None:
        raise ValueError(f'Format must include one of {sorted(FORMAT_BASES)}')
    return {'base': base, 'has_id': has_id, 'spec': s}

def count_lines(path):
    with open(path, 'r') as f:
        return sum(1 for _ in f)


def can_parse_first_k_as_int(tokens, k):
    if len(tokens) < k:
        return False
    try:
        _ = [int(x) for x in tokens[:k]]
        return True
    except Exception:
        return False


def iter_id_rows(file_path, file_format):
    """Yield (source_id, token_list) for supported formats.

    This generator handles the three base formats and respects whether
    each line contains an explicit source ID. Implicit IDs start at zero.
    """
    base = file_format['base']
    has_id = file_format['has_id']

    if base == 'N':
        if has_id:
            with open(file_path, 'r') as f:
                for line in f:
                    tokens = line.split()
                    yield int(tokens[0]), tokens[1:]
            return
        with open(file_path, 'r') as f:
            sid = 0
            for line in f:
                yield sid, line.split()
                sid += 1
        return

    if base == 'B':
        # The blocked layout stores all neighbor-ID lines before all
        # distance lines. Distance lines are ignored in ID-only evaluation.
        with open(file_path, 'r') as f:
            rows = [line.split() for line in f]
        num_lines = len(rows)
        num_points = num_lines // 2
        if num_lines % 2 != 0:
            raise ValueError(f'B format expects even number of lines, got {num_lines} in {file_path}')

        for i in range(num_points):
            id_tokens = rows[i]
            if has_id:
                yield int(id_tokens[0]), id_tokens[1:]
            else:
                yield i, id_tokens
        return

    if base == 'M':
        # Always treat the 'M' base as the paired two-line format.
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                return
            first_tokens = first_line.split()
            if has_id:
                # Each point is represented by an ID line followed by an
                # auxiliary line (e.g., distances). We yield the ID and the
                # token list that follows the ID on the ID line.
                yield int(first_tokens[0]), first_tokens[1:]
                _ = f.readline()
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    yield int(tokens[0]), tokens[1:]
                    _ = f.readline()
            else:
                # No per-line ID: each point occupies two lines; the first
                # line for the first point has already been read.
                sid = 0
                yield sid, first_tokens
                sid += 1
                _ = f.readline()
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    yield sid, tokens
                    sid += 1
                    _ = f.readline()
        return

    raise ValueError(f'Unknown format: {file_format}')


def iter_id_rows_with_dists(file_path, file_format):
    """Yield (source_id, id_tokens_list, dist_tokens_list_or_None).

    This supports the same formats as `iter_id_rows` but attempts to
    return distances when the format uses a two-line representation
    (e.g., base 'B' or our paired 'M'). When distances are not
    present for a format, the third element will be None.
    """
    base = file_format['base']
    has_id = file_format['has_id']

    if base == 'N':
        # Single-line neighbor lists: no distances available
        if has_id:
            with open(file_path, 'r') as f:
                for line in f:
                    tokens = line.split()
                    yield int(tokens[0]), tokens[1:], None
            return
        with open(file_path, 'r') as f:
            sid = 0
            for line in f:
                yield sid, line.split(), None
                sid += 1
        return

    if base == 'B':
        # The blocked layout stores all neighbor-ID lines before all
        # distance lines. Explicit-ID distance rows include a dummy first
        # value, which is discarded to align distances with neighbor IDs.
        with open(file_path, 'r') as f:
            rows = [line.split() for line in f]
        num_lines = len(rows)
        num_points = num_lines // 2
        if num_lines % 2 != 0:
            raise ValueError(f'B format expects even number of lines, got {num_lines} in {file_path}')

        for i in range(num_points):
            id_tokens = rows[i]
            dist_tokens = rows[num_points + i]
            if has_id:
                yield int(id_tokens[0]), id_tokens[1:], dist_tokens[1:]
            else:
                yield i, id_tokens, dist_tokens
        return

    if base == 'M':
        # We always parse the non-interleaved two-line form: an ID/ids line
        # followed by an auxiliary distances line.
        # For explicit IDs, the auxiliary line starts with a dummy value,
        # which is discarded below.
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                return
            first_tokens = first_line.split()
            if has_id:
                # read the following distances line
                dist_line = f.readline()
                dist_tokens = dist_line.split() if dist_line else []
                yield int(first_tokens[0]), first_tokens[1:], dist_tokens[1:]
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    dist_line = f.readline()
                    dist_tokens = dist_line.split() if dist_line else []
                    yield int(tokens[0]), tokens[1:], dist_tokens[1:]
            else:
                sid = 0
                dist_line = f.readline()
                dist_tokens = dist_line.split() if dist_line else []
                yield sid, first_tokens, dist_tokens
                sid += 1
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    dist_line = f.readline()
                    dist_tokens = dist_line.split() if dist_line else []
                    yield sid, tokens, dist_tokens
                    sid += 1
        return

    raise ValueError(f'Unknown format: {file_format}')


def read_ground_truth(ground_truth_file_paths, k, ground_truth_format, use_distance_eval=False, sparse_ground_truth=False):
    # Return a tuple: (gt_id_dict, gt_kth_dist_dict_or_None)
    """Read ground-truth neighbor lists from one or more files.

    Returns a tuple `(gt_id_dict, gt_kth_dist_dict_or_None)`. If
    `use_distance_eval` is True, the second element is a dict mapping
    source IDs to the k-th neighbor distance; otherwise it is None.
    The function validates neighbor count and uniqueness and will
    exit(1) on malformed input to make issues explicit to users.
    """
    gt_data = {}
    gt_kth_dist = {} if use_distance_eval else None
    implicit_ids = not ground_truth_format['has_id']
    for gt_file_path in ground_truth_file_paths:
        print(f'Open {gt_file_path}')
        if ground_truth_format['base'] == 'B':
            num_lines = count_lines(gt_file_path)
            print('Number of lines = ' + str(num_lines))
            print('Number of points = ' + str(num_lines // 2))

        num_points = 0
        if use_distance_eval:
            iterator = iter_id_rows_with_dists
        else:
            iterator = iter_id_rows

        for parsed in iterator(gt_file_path, ground_truth_format):
            if use_distance_eval:
                source_point_id, nn_tokens, dist_tokens = parsed
                # Some datasets store two-line records as [distances][ids]
                # instead of [ids][distances]. In distance-eval mode, detect
                # and auto-correct this to avoid hard failures.
                if dist_tokens is not None:
                    ids_look_valid = can_parse_first_k_as_int(nn_tokens, k)
                    dists_look_like_ids = can_parse_first_k_as_int(dist_tokens, k)
                    if (not ids_look_valid) and dists_look_like_ids:
                        print(
                            f'Warning: swapped ID/distance rows detected for point {source_point_id} in ground truth; auto-correcting')
                        nn_tokens, dist_tokens = dist_tokens, nn_tokens
            else:
                source_point_id, nn_tokens = parsed

            if len(nn_tokens) < k:
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the ground truth is less than {k}, which is {len(nn_tokens)}')
                exit(1)

            try:
                nn_ids = [int(x) for x in nn_tokens[:k]]
            except Exception:
                print(
                    f'Error: cannot parse ground truth neighbor IDs for point {source_point_id}. '
                    f'Check --ground-truth-format (current: {ground_truth_format["spec"]}).')
                exit(1)
            nn_set = set(nn_ids)
            if len(nn_set) != len(nn_ids):
                print(
                    f'Warning: There are duplicate neighbors in point {source_point_id} in the ground truth')
                exit(1)

            if implicit_ids:
                gt_data[num_points] = nn_set
            else:
                if source_point_id in gt_data:
                    print(
                        f'Warning: duplicate source point ID {source_point_id} in the ground truth')
                    exit(1)
                gt_data[source_point_id] = nn_set

            if use_distance_eval:
                # extract k-th neighbor distance if available
                if not dist_tokens or len(dist_tokens) < k:
                    print(
                        f'Error: ground truth distances missing or shorter than k for point {source_point_id}')
                    exit(1)
                try:
                    kth_dist = float(dist_tokens[k - 1])
                except Exception:
                    print(f'Error: cannot parse ground truth distance for point {source_point_id}')
                    exit(1)
                gt_kth_dist[source_point_id] = kth_dist

            num_points += 1

    if not implicit_ids and not sparse_ground_truth:
        if len(gt_data) == 0:
            print('Warning: ground truth is empty')
            exit(1)
        min_id = min(gt_data.keys())
        max_id = max(gt_data.keys())
        if min_id != 0:
            print(f'Warning: missing ground truth entry for point 0')
            exit(1)
        for i in range(0, max_id + 1):
            if i not in gt_data:
                print(f'Warning: missing ground truth entry for point {i}')
                exit(1)

    print('Number of points in ground truth = ' + str(len(gt_data)))
    return gt_data, gt_kth_dist


def evaluate(index_file_path, ground_truth_ids, ground_truth_kth_dist, k, index_format,
sparse_ground_truth=False):
    """Evaluate a single index file against prepared ground truth.

    Returns a tuple `(num_corrects, num_lines, num_skipped)`. If
    `ground_truth_kth_dist` is None an ID-based comparison is used;
    otherwise distance-based evaluation is performed and distances are
    parsed from the index file.
    """
    num_corrects = 0
    num_evaluated_lines = 0
    print(f'Open {index_file_path}')
    num_skipped = 0

    if ground_truth_kth_dist is None:
        # ID-based evaluation (existing behavior)
        for source_point_id, knn_tokens in iter_id_rows(
            index_file_path, index_format):
            if len(knn_tokens) < k:
                print(
                    f'Error: the number of neighbors of point {source_point_id} in the kNN index file is less than {k}, which is {len(knn_tokens)}')
                exit(1)

            knn_ids = [int(x) for x in knn_tokens[:k]]
            knn_ids_set = set(knn_ids)

            # Make sure that the kNN index file contains unique neighbors
            if len(knn_ids_set) != len(knn_ids):
                print(
                    f'Error: There are duplicate neighbors in point {source_point_id}')
                for nid in knn_ids:
                    if nid not in knn_ids_set:
                        print(f'  Duplicate neighbor ID: {nid}')
                    else:
                        knn_ids_set.remove(nid)
                exit(1)

            if source_point_id not in ground_truth_ids:
                if sparse_ground_truth:
                    num_skipped += 1
                    continue

                print(
                    f'Error: missing ground truth for point {source_point_id}')
                exit(1)

            ground_truth_knn_ids = ground_truth_ids[source_point_id]
            if len(ground_truth_knn_ids) < k:
                print(
                    f'Error: the number of neighbors of point {source_point_id} in the ground truth is less than {k}, which is {len(ground_truth_knn_ids)}')
                exit(1)

            num_corrects += len(knn_ids_set & ground_truth_knn_ids)
            num_evaluated_lines += 1
    else:
        # Distance-based evaluation: do not check IDs. A neighbor is correct if
        # its distance <= ground-truth k-th neighbor distance for that source.
        for source_point_id, knn_ids, knn_dists in iter_id_rows_with_dists(
            index_file_path, index_format):
            if len(knn_ids) < k:
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the kNN index file is less than {k}, which is {len(knn_ids)}')
                exit(1)

            if knn_dists is None or len(knn_dists) < k:
                print(f'Error: index file does not contain distances required for distance-based evaluation for point {source_point_id}')
                exit(1)

            if source_point_id not in ground_truth_kth_dist:
                if sparse_ground_truth:
                    num_skipped += 1
                    continue
                print(
                    f'Error: missing ground truth distance for point {source_point_id}')
                exit(1)

            try:
                threshold = float(ground_truth_kth_dist[source_point_id])
            except Exception:
                print(f'Error: cannot parse ground truth k-th distance for point {source_point_id}')
                exit(1)

            # Count neighbors whose distance is <= threshold
            corrected = 0
            for d in knn_dists[:k]:
                try:
                    dv = float(d)
                except Exception:
                    print(f'Error: cannot parse index neighbor distance for point {source_point_id}')
                    exit(1)
                if dv <= threshold:
                    corrected += 1
            num_corrects += corrected
            num_evaluated_lines += 1

    return num_corrects, num_evaluated_lines, num_skipped


def main():
    """Main entry point: parse args, read ground truth, and evaluate.

    Prints a short summary including recall and timing information.
    """
    args = parse_args()
    print('Arguments:')
    print(args)

    print('Read ground truth')
    gt_files = find_files(args.ground_truth_file_path)
    if len(gt_files) > 1 and not args.ground_truth_format['has_id']:
        print('Error: multiple ground truth files require IDs in the file format')
        exit(1)
    ground_truth_ids, ground_truth_kth_dist = read_ground_truth(
        gt_files, args.k, args.ground_truth_format, use_distance_eval=args.use_distance_eval, sparse_ground_truth=args.sparse_ground_truth)

    index_files = find_files(args.index_file_path)
    if len(index_files) > 1 and not args.index_format['has_id']:
        print('Error: multiple index files require IDs in the file format')
        exit(1)
    print('Found ' + str(len(index_files)) + ' index files')

    num_evaluated_lines = 0
    num_corrects = 0
    num_skipped = 0
    start_time = time.time()
    for index_file in index_files:
        corrects, evaluated_lines, skipped_lines = evaluate(
            index_file, ground_truth_ids, ground_truth_kth_dist, args.k, args.index_format, sparse_ground_truth=args.sparse_ground_truth)
        num_corrects += corrects
        num_evaluated_lines += evaluated_lines
        num_skipped += skipped_lines
    end_time = time.time()

    print(f'Number of points in test index = {num_evaluated_lines + num_skipped}')
    print(f'Number of skipped test points = {num_skipped}')
    print(f'Number of total points in ground truth = {len(ground_truth_ids)}')
    print(f'Number of evaluated test neighbors = {num_evaluated_lines * args.k}')
    print(f'Number of correct test neighbors = {num_corrects}')
    print(f'Recall Score = {num_corrects / (num_evaluated_lines * args.k)}')
    print(f'Time = {end_time - start_time}')


if __name__ == '__main__':
    main()
