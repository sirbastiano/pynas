import numpy as np
from skimage.draw import line

def radial_sampling(matrix: np.ndarray, center_x: int, center_y: int, S: int, n: int) -> list:
    """
    Extracts pixel values along n equidistant radial lines from the center of a SxS window in the matrix.

    Args:
        matrix (np.ndarray): 2D input matrix.
        center_x (int): X-coordinate of the central point.
        center_y (int): Y-coordinate of the central point.
        S (int): Size of the square window (must be odd).
        n (int): Number of radial lines.

    Returns:
        list: A list of n arrays, each containing values along one radial line.
    """
    if S % 2 == 0:
        raise ValueError("Window size S must be odd.")

    half_S = S // 2
    
    # Define the window boundaries
    start_y = center_y - half_S
    end_y = center_y + half_S + 1
    start_x = center_x - half_S
    end_x = center_x + half_S + 1

    if (start_y < 0 or end_y > matrix.shape[0] or 
        start_x < 0 or end_x > matrix.shape[1]):
        raise ValueError("Window exceeds matrix bounds.")

    window = matrix[start_y:end_y, start_x:end_x]
    center = half_S
    radius = half_S

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    samples = []

    for theta in angles:
        end_y = int(center + radius * np.sin(theta))
        end_x = int(center + radius * np.cos(theta))

        rr, cc = line(center, center, end_y, end_x)
        rr = np.clip(rr, 0, S-1)
        cc = np.clip(cc, 0, S-1)
        values = window[rr, cc]
        samples.append(values)

    return samples

# Example usage
if __name__ == "__main__":
    mat = np.random.rand(101, 101)
    cx, cy = 50, 50
    lines = radial_sampling(mat, cx, cy, S=21, n=8)
    for i, line_vals in enumerate(lines):
        print(f"Line {i}: {line_vals}")
