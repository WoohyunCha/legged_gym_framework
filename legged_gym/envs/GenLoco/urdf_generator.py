import xml.etree.ElementTree as ET
import numpy as np
import numpy as np
from stl import mesh
import os

class urdf_generator():
    def __init__(self, config) -> None:
        pass
    

    def create_link(self, name, length, mass, inertia):
        if name == "world":
            link = ET.Element("link", name=name)
        
        elif name == "base_link":
            # Create base link mesh
            width = length["width"]
            height = length["height"]
            depth = length["depth"]
            
            vertices = np.array([
                [0, 0, 0],            # Vertex 1
                [depth, 0, 0],         # Vertex 2
                [depth, width, 0],    # Vertex 3
                [0, width, 0],        # Vertex 4
                [0, 0, height],        # Vertex 5
                [depth, 0, height],     # Vertex 6
                [depth, width, height],# Vertex 7
                [0, width, height]     # Vertex 8
            ])

            # Define faces of the rectangular shape using vertices
            faces = np.array([
                [0, 3, 7], [0, 7, 4],  # Bottom
                [1, 2, 6], [1, 6, 5],  # Top
                [0, 4, 5], [0, 5, 1],  # Side 1
                [2, 3, 7], [2, 7, 6],  # Side 2
                [0, 1, 2], [0, 2, 3],  # Side 3
                [4, 5, 6], [4, 6, 7]   # Side 4
            ])

            # Create STL mesh
            rect_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, face in enumerate(faces):
                for j in range(3):
                    rect_mesh.vectors[i][j] = vertices[face[j]]            
            rect_mesh.save(os.path.join("..", "..", "..", "resources", "robots", "GenLoco", "meshes", "base_link.stl"))
            
            
            link = ET.Element("link", name=name)
            
            visual = ET.SubElement(link, "visual")        
            ET.SubElement(visual, "origin", xyz=str(depth/2)+" "+str(width/2)+" "+str(height/2), rpy="0 0 0")  
            ET.SubElement(ET.SubElement(visual, "geomtry"), "mesh", filename=str(os.path.join("..", "..", "..", "resources", "robots", "GenLoco", "meshes", "base_link.stl")))    
            
            inertia = ET.SubElement(link, "inertial")
            ET.SubElement(inertia, "mass", value=str(mass))
            ET.SubElement(inertia, "inertia", ixx=str(inertia["ixx"]), ixy=str(inertia["ixy"]),ixz=str(inertia["ixz"]),iyy=str(inertia["iyy"]),iyz=str(inertia["iyz"]),izz=str(inertia["izz"]))
            ET.SubElement(inertia, "origin", xyz=str(depth/2)+" "+str(width/2)+" "+str(height/2), rpy="0 0 0") # This should be changed to a realistic value
            
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", xyz=str(depth/2)+" "+str(width/2)+" "+str(height/2), rpy="0 0 0")
            ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=str(os.path.join("..", "..", "..", "resources", "robots", "GenLoco", "meshes", "base_link.stl")))
        
        else:
            r_length = length["length"]
            r_radius = length["radius"]
            link = ET.Element("link", name=name)
            
            visual = ET.SubElement(link, "visual")        
            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")  
            ET.SubElement(ET.SubElement(visual, "geomtry"), "cylinder", length=str(r_length), radius=str(r_radius))    
            
            inertia = ET.SubElement(link, "inertial")
            ET.SubElement(inertia, "mass", value=str(mass))
            ET.SubElement(inertia, "inertia", ixx=str(inertia["ixx"]), ixy=str(inertia["ixy"]),ixz=str(inertia["ixz"]),iyy=str(inertia["iyy"]),iyz=str(inertia["iyz"]),izz=str(inertia["izz"]))
            ET.SubElement(inertia, "origin", xyz="0 0 0", rpy="0 0 0") # This should be changed to a realistic value
            
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", length=str(r_length), radius=str(r_radius))
        
        return link

    def create_joint(self, name, joint_type, parent, child, axis="0 1 0", origin="0 0 0", limit = {"effort": 200, "velocity": 1., "lower": -3.14, "upper": 3.14}):
        joint = ET.Element("joint", name=name, type=joint_type)

        parent_element = ET.SubElement(joint, "parent", link=parent)
        child_element = ET.SubElement(joint, "child", link=child)

        ET.SubElement(joint, "axis", xyz=axis)
        ET.SubElement(joint, "origin", xyz=origin)

        return joint

    def generate_urdf(self):
        
        sz_scale = np.random.uniform(0.7, 1.3)
        mass_scale = sz_scale**3
        inertia_scale = sz_scale**5
        ref_length = 0.5 # This is the length of the upper leg when sz_scale is 1. Every other length values are relative to this value.
        ref_radius = 0.1 # This is the radius of the upper leg when sz_scale is 1. Every other radius values are relative to this value.
        
        ref_length *= sz_scale
        ref_radius *= sz_scale
        
        density_range = (50, 100) # Must be modified
        
        robot = ET.Element("robot", name="GenLoco")

        # World link
        robot.append(self.create_link(name="world"))
        
        # Virtual joint
        robot.append(self.create_joint(name="virtual", joint_type="floating", parent="world", child="base_link"))
        
        # Base link
        '''
        Length is determined with respect to ref_radius and ref_length, which are the radius and length of the cylindrical upper leg link.
        The range for np.random.uniform can be modified.
        Mass is determined with respect to the volume of the link and a randomized density. Range of density can be modified.
        Inerteia is determined with respect to the mass and a randomized inertia factor. 
        '''
        length = {"width" : 4 * ref_radius * np.random.uniform(1.2, 2), "depth" : 2 * ref_radius * np.random.uniform(1.0, 3.), "height" : ref_length * np.random.uniform(0.5, 1.5)}
        mass = 8 * ref_radius**2 * ref_length * np.random.unform(*density_range)
        inertia = {"ixx": mass * , "iyy": 0, "izz": 0, "ixy": 0, "ixz": 0, "iyz": 0}
        
        base_link = self.create_link("base_link", length=length)
        robot.append(base_link)

        # Left leg
        left_upper_leg = create_link("left_upper_leg")
        left_lower_leg = create_link("left_lower_leg")
        left_foot = create_link("left_foot")

        robot.extend([left_upper_leg, left_lower_leg, left_foot])

        robot.append(create_joint("spherical_joint_left", "spherical", "base_link", "left_upper_leg"))

        robot.append(create_joint("pitch_joint_left_upper", "revolute", "left_upper_leg", "left_lower_leg", axis="1 0 0"))
        robot.append(create_joint("pitch_joint_left_lower", "revolute", "left_lower_leg", "left_foot", axis="1 0 0"))

        # Right leg
        right_upper_leg = create_link("right_upper_leg")
        right_lower_leg = create_link("right_lower_leg")
        right_foot = create_link("right_foot")

        robot.extend([right_upper_leg, right_lower_leg, right_foot])

        robot.append(create_joint("spherical_joint_right", "spherical", "base_link", "right_upper_leg"))

        robot.append(create_joint("pitch_joint_right_upper", "revolute", "right_upper_leg", "right_lower_leg", axis="1 0 0"))
        robot.append(create_joint("pitch_joint_right_lower", "revolute", "right_lower_leg", "right_foot", axis="1 0 0"))

        # Create URDF string
        urdf_str = ET.tostring(robot).decode("utf-8")
        return urdf_str


    def create_rectangular_mesh(self, width, length, height):
        # Create vertices of the rectangular shape


        return rect_mesh
