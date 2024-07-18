import xml.etree.ElementTree as ET
import numpy as np
import numpy as np
import os
import yaml

class urdf_generator():
    def __init__(self):
        with open('cfg/morphology_range.yaml') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.size = config['size']
        self.base_density_range = config['base_density_range']
        self.nominal_base_length = config['nominal_base_length']
        self.nominal_base_width = config['nominal_base_width']
        self.nominal_base_height = config['nominal_base_height']
        self.nominal_thigh_mass = config['nominal_thigh_mass']
        self.nominal_thigh_length = config['nominal_thigh_length']
        self.nominal_thigh_radius = config['nominal_thigh_radius']
        self.nominal_shin_mass = config['nominal_shin_mass']
        self.shin_length_range= config['shin_length_range']
        self.shin_radius_range = config['shin_radius_range']
        self.nominal_foot_mass = config['nominal_foot_mass']
        self.foot_height_range = config['foot_height_range']
        self.foot_length_range = config['foot_length_range']
        self.foot_width_range = config['foot_width_range']
        self.link_mass_range = config['link_mass_range']
        self.link_inertia_range = config['link_inertia_range']
        self.link_length_range = config['link_length_range']
        self.link_com_position_range = config['link_com_position_range']

    def _create_link(self, name, length, mass, inertia, origin={}):
        if name == "base_link":
            # Create base link mesh
            print("Base link created")
            print("mass : ", mass)
            width = np.round(length["width"], 3)
            height = np.round(length["height"], 3)
            depth = np.round(length["depth"], 3)
            
            link = ET.Element("link", name=name)
            
            visual = ET.SubElement(link, "visual")        
            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")  
            ET.SubElement(ET.SubElement(visual, "geometry"), "box", size=str(depth)+" "+str(width)+" "+str(height))
            ET.SubElement( ET.SubElement(visual, "material", name="grey"), "color", rgba="0.9 0.9 0.9 1.0")
            
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value=str(mass))
            ET.SubElement(inertial, "inertia", ixx=str(inertia["ixx"]), ixy=str(inertia["ixy"]),ixz=str(inertia["ixz"]),iyy=str(inertia["iyy"]),iyz=str(inertia["iyz"]),izz=str(inertia["izz"]))
            ET.SubElement(inertial, "origin", xyz=str(depth * np.random.uniform(*self.link_com_position_range))+" "+str(width * np.random.uniform(*self.link_com_position_range))+" "+str(height * np.random.uniform(*self.link_com_position_range )), rpy="0 0 0")
            
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=str(depth)+" "+str(width)+" "+str(height))
        else:
            if mass == 0:
                print("Virtual link created")
                link = ET.Element("link", name=name)
                inertia = ET.SubElement(link, "inertial")
                ET.SubElement(inertia, "mass", value="1.e-4")
                ET.SubElement(inertia, "inertia", ixx="1.e-7", ixy="0",ixz="0",iyy="1.e-7",iyz="0",izz="1.e-7")
                
            else:
                # Cylindrical link
                # if origin == {}:
                #     print("ORDINARY LINK CANNOT HAVE EMPTY ORIGIN")
                #     exit()    
                # r_length = length["length"]
                # r_radius = length["radius"]
                # link = ET.Element("link", name=name)
                
                # visual = ET.SubElement(link, "visual")        
                # ET.SubElement(visual, "origin", xyz=str(origin["x"])+" "+str(origin["y"])+" "+str(origin["z"]), rpy="0 0 0")  
                # ET.SubElement(ET.SubElement(visual, "geometry"), "cylinder", length=str(r_length), radius=str(r_radius))    
                
                # inertial = ET.SubElement(link, "inertial")
                # ET.SubElement(inertial, "mass", value=str(mass))
                # ET.SubElement(inertial, "inertia", ixx=str(inertia["ixx"]), ixy=str(inertia["ixy"]),ixz=str(inertia["ixz"]),iyy=str(inertia["iyy"]),iyz=str(inertia["iyz"]),izz=str(inertia["izz"]))
                # ET.SubElement(inertial, "origin", xyz=str(origin["x"])+" "+str(origin["y"])+" "+str(origin["z"]), rpy="0 0 0") # This should be changed to a realistic value
                
                # collision = ET.SubElement(link, "collision")
                # ET.SubElement(collision, "origin", xyz=str(origin["x"])+" "+str(origin["y"])+" "+str(origin["z"]), rpy="0 0 0")
                # ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", length=str(r_length), radius=str(r_radius))
                print(name + " link created")
                print("mass : ", mass)
                if origin == {}:
                    print("ORDINARY LINK CANNOT HAVE EMPTY ORIGIN")
                    exit()    
                width = length["width"]
                height = length["height"]
                depth = length["depth"]
                
                
                link = ET.Element("link", name=name)
                
                visual = ET.SubElement(link, "visual")        
                ET.SubElement(visual, "origin", xyz=str(origin["x"])+" "+str(origin["y"])+" "+str(origin["z"]), rpy="0 0 0")  
                ET.SubElement(ET.SubElement(visual, "geometry"), "box", size=str(depth)+" "+str(width)+" "+str(height))    
                
                inertial = ET.SubElement(link, "inertial")
                ET.SubElement(inertial, "mass", value=str(mass))
                ET.SubElement(inertial, "inertia", ixx=str(inertia["ixx"]), ixy=str(inertia["ixy"]),ixz=str(inertia["ixz"]),iyy=str(inertia["iyy"]),iyz=str(inertia["iyz"]),izz=str(inertia["izz"]))
                ET.SubElement(inertial, "origin", xyz=str(origin["x"])+" "+str(origin["y"])+" "+str(origin["z"]), rpy="0 0 0")
                
                collision = ET.SubElement(link, "collision")
                ET.SubElement(collision, "origin", xyz=str(origin["x"])+" "+str(origin["y"])+" "+str(origin["z"]), rpy="0 0 0")
                ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=str(depth)+" "+str(width)+" "+str(height))
        
        return link

    def _create_joint(self, name, joint_type, parent, child, axis="0 1 0", origin={"x":0, "y":0, "z":0}, limit = {"effort": 200, "velocity": 1., "lower": -3.14, "upper": 3.14}):
        joint = ET.Element("joint", name=name, type=joint_type)

        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "limit", effort=str(limit["effort"]), velocity=str(limit["velocity"]), lower=str(limit["lower"]), upper=str(limit["upper"]))
        ET.SubElement(joint, "axis", xyz=axis)
        ET.SubElement(joint, "origin", xyz=str(origin["x"])+" "+str(origin["y"])+" "+str(origin["z"]))

        return joint

    def generate_urdf(self):
        
        #### For info logging
        robot_mass = 0
        base_mass = 0
        thigh_mass = 0
        shin_mass = 0
        foot_mass = 0
        ###

        sz_scale = np.random.uniform(*self.size)
        mass_scale = sz_scale**3
        inertia_scale = sz_scale**5
        thigh_length = sz_scale*np.random.uniform(*self.link_length_range)*self.nominal_thigh_length # This is the length of the upper leg when sz_scale is 1. Every other length values are relative to this value.
        thigh_radius = sz_scale*np.random.uniform(*self.link_length_range)*self.nominal_thigh_radius # This is the radius of the upper leg when sz_scale is 1. Every other radius values are relative to this value.
        shin_length = thigh_length*np.random.uniform(*self.shin_length_range)
        shin_radius = thigh_length * np.random.uniform(*self.shin_radius_range)
        foot_length = thigh_length*np.random.uniform(*self.foot_length_range)
        foot_width = thigh_length*np.random.uniform(*self.foot_width_range)
        foot_height = thigh_length*np.random.uniform(*self.foot_height_range)
        base_width = sz_scale*np.random.uniform(*self.link_length_range)*self.nominal_base_width
        base_height = sz_scale*np.random.uniform(*self.link_length_range)*self.nominal_base_height
        base_length = sz_scale*np.random.uniform(*self.link_length_range)*self.nominal_base_length
        
        
        robot = ET.Element("robot", name="GenLoco")
        # Base link
        '''
        Length is determined with respect to ref_radius and ref_length, which are the radius and length of the cylindrical upper leg link.
        The range for np.random.uniform can be modified.
        Mass is determined with respect to the volume of the link and a randomized density. Range of density can be modified.
        Inerteia is determined with respect to the mass and a randomized inertia factor. 
        '''
        length = {"width" : base_width, "depth" : base_length, "height" : base_height}
        mass = base_width*base_height*base_length * np.random.uniform(*self.base_density_range)
        # I dont know how I should set the inertia
        inertia = {"ixx": mass * (length["width"]**2 + length["height"]**2) * 1.e-2 * np.random.uniform(*self.link_inertia_range), "iyy": mass * (length["depth"]**2 + length["height"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "izz": mass * (length["width"]**2 + length["depth"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "ixy": mass * (length["width"]*length["depth"]) * 1.e-4, "ixz": mass * (length["height"]*length["depth"]) * 1.e-4, "iyz": mass * (length["height"]*length["width"]) * 1.e-4}
        
        base_link = self._create_link("base_link", length=length, mass=mass, inertia=inertia)
        robot.append(base_link)
        robot_mass += mass
        base_mass = mass

        # Left Hip
        '''
        There is no spherical joint in URDF.
        Therefore, we must use three revolute joints with dummy links between.
        Dummy links will have small mass and inertia https://answers.ros.org/question/258420/ball-joint-in-urdf/
        '''
        origin = {"x": 0, "y": length["width"] * 0.5, "z": -length["height"] * 0.5}
        robot.append(self._create_joint(name="L_HipYaw", joint_type="revolute", parent="base_link", child="L_HipYaw", axis="0 0 1", origin=origin))
        origin = {"x": 0, "y": 0, "z": 0}
        robot.append(self._create_link("L_HipYaw", length={}, mass=0, inertia={})) # mass = 0 create dummy link of near zero mass and inertia
        robot.append(self._create_joint(name="L_HipRoll", joint_type="revolute", parent="L_HipYaw", child="L_HipRoll", axis="1 0 0"))
        robot.append(self._create_link("L_HipRoll", length={}, mass=0, inertia={})) # mass = 0 create dummy link of near zero mass and inertia
        robot.append(self._create_joint(name="L_HipPitch", joint_type="revolute", parent="L_HipRoll", child="L_HipPitch", axis="0 1 0"))
        # Right Hip
        origin = {"x": 0, "y": -length["width"] * 0.5, "z": -length["height"] * 0.5}
        robot.append(self._create_joint(name="R_HipYaw", joint_type="revolute", parent="base_link", child="R_HipYaw", axis="0 0 1", origin=origin))
        origin = {"x": 0, "y": 0, "z": 0}
        robot.append(self._create_link("R_HipYaw", length={}, mass=0, inertia={})) # mass = 0 create dummy link of near zero mass and inertia
        robot.append(self._create_joint(name="R_HipRoll", joint_type="revolute", parent="R_HipYaw", child="R_HipRoll", axis="1 0 0"))
        robot.append(self._create_link("R_HipRoll", length={}, mass=0, inertia={})) # mass = 0 create dummy link of near zero mass and inertia
        robot.append(self._create_joint(name="R_HipPitch", joint_type="revolute", parent="R_HipRoll", child="R_HipPitch", axis="0 1 0"))

        # Upper legs
        length = {"width" : thigh_radius, "depth" : thigh_radius, "height" : thigh_length}
        origin = {"x": 0, "y": 0, "z": -sz_scale*self.nominal_thigh_length*0.5} # Since length is randomized, setting com position fixed will lead to random com position
        mass = mass_scale * self.nominal_thigh_mass * np.random.uniform(*self.link_mass_range)
        # I dont know how I should set the inertia
        inertia = {"ixx": mass * (length["width"]**2 + length["height"]**2) * 1.e-2 * np.random.uniform(*self.link_inertia_range), "iyy": mass * (length["depth"]**2 + length["height"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "izz": mass * (length["width"]**2 + length["depth"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "ixy": mass * (length["width"]*length["depth"]) * 1.e-4, "ixz": mass * (length["height"]*length["depth"]) * 1.e-4, "iyz": mass * (length["height"]*length["width"]) * 1.e-4}
   
        robot.append(self._create_link("L_HipPitch", length=length, mass=mass, inertia=inertia, origin=origin))
        origin = {"x": 0, "y": 0, "z": -length["height"]}
        robot.append(self._create_joint(name="L_KneePitch", joint_type="revolute", parent="L_HipPitch", child="L_KneePitch", axis="0 1 0", origin=origin))        
        robot_mass += 2*mass
        thigh_mass = mass
        # Upper legs
        origin = {"x": 0, "y": 0, "z": -sz_scale*self.nominal_thigh_length*0.5}
        # I dont know how I should set the inertia

        robot.append(self._create_link("R_HipPitch", length=length, mass=mass, inertia=inertia, origin=origin))
        origin = {"x": 0, "y": 0, "z": -length["height"]}        
        robot.append(self._create_joint(name="R_KneePitch", joint_type="revolute", parent="R_HipPitch", child="R_KneePitch", axis="0 1 0", origin=origin))        
        
        # Lower legs
        length = {"width" : shin_radius, "depth": shin_radius, "height" : shin_length}
        # origin = {"x": 0, "y": 0, "z": -shin_length*0.5*np.random.uniform(*self.link_com_position_range)}
        origin = {"x": 0, "y": 0, "z": -shin_length*0.5}
        mass = mass_scale * self.nominal_shin_mass*np.random.uniform(*self.link_mass_range)
        robot_mass += 2*mass
        shin_mass = mass
        # I dont know how I should set the inertia
        inertia = {"ixx": mass * (length["width"]**2 + length["height"]**2) * 1.e-2 * np.random.uniform(*self.link_inertia_range), "iyy": mass * (length["depth"]**2 + length["height"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "izz": mass * (length["width"]**2 + length["depth"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "ixy": mass * (length["width"]*length["depth"]) * 1.e-4, "ixz": mass * (length["height"]*length["depth"]) * 1.e-4, "iyz": mass * (length["height"]*length["width"]) * 1.e-4}
 
        robot.append(self._create_link("L_KneePitch", length=length, mass=mass, inertia=inertia, origin=origin))        
        origin = {"x": 0, "y": 0, "z": -length["height"]}
        robot.append(self._create_joint(name="L_AnklePitch", joint_type="revolute", parent="L_KneePitch", child="L_Foot", axis="0 1 0", origin=origin))        

        # Lower legs
        # I dont know how I should set the inertia
        # origin = {"x": 0, "y": 0, "z": -shin_length*0.5*np.random.uniform(*self.link_com_position_range)}
        origin = {"x": 0, "y": 0, "z": -shin_length*0.5}

        robot.append(self._create_link("R_KneePitch", length=length, mass=mass, inertia=inertia, origin=origin))        
        origin = {"x": 0, "y": 0, "z": -length["height"]}
        robot.append(self._create_joint(name="R_AnklePitch", joint_type="revolute", parent="R_KneePitch", child="R_Foot", axis="0 1 0", origin=origin))        
        
        # Foot        
        length = {"width" : foot_width, "depth" : foot_length, "height" : foot_height}
        origin = {"x": 0, "y": 0, "z": -length["height"]*0.5}
        mass = mass_scale * self.nominal_foot_mass*np.random.uniform(*self.link_mass_range)
        inertia = {"ixx": mass * (length["width"]**2 + length["height"]**2) * 1.e-2 * np.random.uniform(*self.link_inertia_range), "iyy": mass * (length["depth"]**2 + length["height"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "izz": mass * (length["width"]**2 + length["depth"]**2) * 1.e-2* np.random.uniform(*self.link_inertia_range), "ixy": mass * (length["width"]*length["depth"]) * 1.e-4, "ixz": mass * (length["height"]*length["depth"]) * 1.e-4, "iyz": mass * (length["height"]*length["width"]) * 1.e-4}
        robot_mass += 2*mass
        foot_mass = mass
        robot.append(self._create_link("L_Foot", length=length, mass=mass, inertia=inertia, origin=origin)) 
        robot.append(self._create_link("R_Foot", length=length, mass=mass, inertia=inertia, origin=origin))        

        # Create URDF file
        indent(robot)
        tree = ET.ElementTree(robot)
        tree.write(os.path.join("GenLoco.urdf"))
        
        info = {
            "robot mass": robot_mass,
            "thigh mass": thigh_mass,
            "shin mass": shin_mass,
            "foot mass": foot_mass
        }
        
        return tree, info
    
    
###########HELPER#############
def indent(elem, level=0): #https://goo.gl/J8VoDK
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            