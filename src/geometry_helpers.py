import numpy as np

def derive_angular_singularities_params(singular_points, integration_center):
      shifted_points = [point - integration_center for point in singular_points if np.linalg.norm(point-integration_center) > 1e-13]
      line_vecs = [shifted_point/np.linalg.norm(shifted_point) for shifted_point in shifted_points ]
      line_angles = [
            np.array([
                  np.arccos(line_vec[2]),
                  np.atan2(line_vec[1], line_vec[0]) if np.atan2(line_vec[1], line_vec[0]) >=0 else 2*np.pi+np.atan2(line_vec[1], line_vec[0])
            ]) for line_vec in line_vecs
      ]

      # account for periodicity
      line_sings_angle_periodic = [
            periodic
            for angle in line_angles  
            for periodic in [
                  angle,
                  angle + np.array([np.pi, 0]),
                  angle - np.array([np.pi, 0]),
                  angle + np.array([0, 2*np.pi]),
                  angle - np.array([0, 2*np.pi]),
            ]
      ]

      point_sphericals = [
            [np.linalg.norm(shifted_points[i]), angle[0], angle[1]] for i,angle in enumerate(line_angles)     
      ]

      point_sings_periodic = [
            periodic
            for point in point_sphericals
            for periodic in [
                  point,
                  point + np.array([0, np.pi, 0]),
                  point - np.array([0, np.pi, 0]),
                  point + np.array([0, 0, 2*np.pi]),
                  point - np.array([0, 0, 2*np.pi]),
            ]
      ]

      return line_sings_angle_periodic, point_sings_periodic