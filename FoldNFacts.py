import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
import py3Dmol
import requests
import biotite.structure.io as bsio
import seaborn as sns
from stmol import showmol
import io

#background colour along with the sidebar
preferred_color = "#C5ADC5"

st.markdown(
    """
    <style>
    /* Main app background with gradient */
    .stApp {
        background: linear-gradient(to bottom right, #C5ADC5, #B2B5E0);
        background-attachment: fixed;
    }

    /* Sidebar background with same gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom right, #C5ADC5, #B2B5E0);
    }

    /* Optional: remove box shadow for cleaner look */
    section[data-testid="stSidebar"] > div:first-child {
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Add a black border on the right of the sidebar */
    section[data-testid="stSidebar"] {
        border-right: 3px solid black;
        padding-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Amino acid properties data
AA_DATA = {
    'A': {'name': 'Alanine', 'weight': 89.09, 'hydropathy': 1.8, 'pKa': 2.34, 'pKb': 9.69, 'pKx': None, 'pi': 6.0, 'class': 'nonpolar'},
    'R': {'name': 'Arginine', 'weight': 174.2, 'hydropathy': -4.5, 'pKa': 2.17, 'pKb': 9.04, 'pKx': 12.48, 'pi': 10.76, 'class': 'positive'},
    'N': {'name': 'Asparagine', 'weight': 132.12, 'hydropathy': -3.5, 'pKa': 2.02, 'pKb': 8.8, 'pKx': None, 'pi': 5.41, 'class': 'polar'},
    'D': {'name': 'Aspartic acid', 'weight': 133.1, 'hydropathy': -3.5, 'pKa': 1.88, 'pKb': 9.6, 'pKx': 3.65, 'pi': 2.77, 'class': 'negative'},
    'C': {'name': 'Cysteine', 'weight': 121.16, 'hydropathy': 2.5, 'pKa': 1.96, 'pKb': 10.28, 'pKx': 8.18, 'pi': 5.07, 'class': 'polar'},
    'E': {'name': 'Glutamic acid', 'weight': 147.13, 'hydropathy': -3.5, 'pKa': 2.19, 'pKb': 9.67, 'pKx': 4.25, 'pi': 3.22, 'class': 'negative'},
    'Q': {'name': 'Glutamine', 'weight': 146.15, 'hydropathy': -3.5, 'pKa': 2.17, 'pKb': 9.13, 'pKx': None, 'pi': 5.65, 'class': 'polar'},
    'G': {'name': 'Glycine', 'weight': 75.07, 'hydropathy': -0.4, 'pKa': 2.34, 'pKb': 9.6, 'pKx': None, 'pi': 5.97, 'class': 'nonpolar'},
    'H': {'name': 'Histidine', 'weight': 155.16, 'hydropathy': -3.2, 'pKa': 1.82, 'pKb': 9.17, 'pKx': 6.0, 'pi': 7.59, 'class': 'positive'},
    'I': {'name': 'Isoleucine', 'weight': 131.17, 'hydropathy': 4.5, 'pKa': 2.36, 'pKb': 9.6, 'pKx': None, 'pi': 6.02, 'class': 'nonpolar'},
    'L': {'name': 'Leucine', 'weight': 131.17, 'hydropathy': 3.8, 'pKa': 2.36, 'pKb': 9.6, 'pKx': None, 'pi': 5.98, 'class': 'nonpolar'},
    'K': {'name': 'Lysine', 'weight': 146.19, 'hydropathy': -3.9, 'pKa': 2.18, 'pKb': 8.95, 'pKx': 10.53, 'pi': 9.74, 'class': 'positive'},
    'M': {'name': 'Methionine', 'weight': 149.21, 'hydropathy': 1.9, 'pKa': 2.28, 'pKb': 9.21, 'pKx': None, 'pi': 5.74, 'class': 'nonpolar'},
    'F': {'name': 'Phenylalanine', 'weight': 165.19, 'hydropathy': 2.8, 'pKa': 1.83, 'pKb': 9.13, 'pKx': None, 'pi': 5.48, 'class': 'nonpolar'},
    'P': {'name': 'Proline', 'weight': 115.13, 'hydropathy': -1.6, 'pKa': 1.99, 'pKb': 10.6, 'pKx': None, 'pi': 6.3, 'class': 'nonpolar'},
    'S': {'name': 'Serine', 'weight': 105.09, 'hydropathy': -0.8, 'pKa': 2.21, 'pKb': 9.15, 'pKx': None, 'pi': 5.68, 'class': 'polar'},
    'T': {'name': 'Threonine', 'weight': 119.12, 'hydropathy': -0.7, 'pKa': 2.09, 'pKb': 9.1, 'pKx': None, 'pi': 5.87, 'class': 'polar'},
    'W': {'name': 'Tryptophan', 'weight': 204.23, 'hydropathy': -0.9, 'pKa': 2.83, 'pKb': 9.39, 'pKx': None, 'pi': 5.89, 'class': 'nonpolar'},
    'Y': {'name': 'Tyrosine', 'weight': 181.19, 'hydropathy': -1.3, 'pKa': 2.2, 'pKb': 9.11, 'pKx': 10.07, 'pi': 5.66, 'class': 'polar'},
    'V': {'name': 'Valine', 'weight': 117.15, 'hydropathy': 4.2, 'pKa': 2.32, 'pKb': 9.62, 'pKx': None, 'pi': 5.96, 'class': 'nonpolar'}
}

#Protein Properties
def calculate_protein_properties(sequence):
    sequence = sequence.upper().strip()
    valid_aas = set(AA_DATA.keys())
    invalid_chars = set(sequence) - valid_aas
    if invalid_chars:
        raise ValueError(f"Invalid amino acid characters: {', '.join(invalid_chars)}")
    
    length = len(sequence)
    aa_counts = Counter(sequence)
    molecular_weight = sum(AA_DATA[aa]['weight'] for aa in sequence) - ((length - 1) * 18.015)
    composition = {aa: (count / length * 100) for aa, count in aa_counts.items()}
    avg_hydropathy = sum(AA_DATA[aa]['hydropathy'] for aa in sequence) / length

    def calculate_charge(pH=7.0):
        positive, negative = 0, 0
        for aa in sequence:
            if aa in ['H', 'K', 'R'] and AA_DATA[aa]['pKx'] and pH < AA_DATA[aa]['pKx']:
                positive += 1
            if aa == 'D' and pH > AA_DATA['D']['pKx']:
                negative += 1
            elif aa == 'E' and pH > AA_DATA['E']['pKx']:
                negative += 1
            elif aa == 'C' and pH > AA_DATA['C']['pKx']:
                negative += 1
            elif aa == 'Y' and pH > AA_DATA['Y']['pKx']:
                negative += 1
        return positive - negative

    net_charge = calculate_charge()
    class_counts = {
        'positive': sum(1 for aa in sequence if AA_DATA[aa]['class'] == 'positive'),
        'negative': sum(1 for aa in sequence if AA_DATA[aa]['class'] == 'negative'),
        'polar': sum(1 for aa in sequence if AA_DATA[aa]['class'] == 'polar'),
        'nonpolar': sum(1 for aa in sequence if AA_DATA[aa]['class'] == 'nonpolar')
    }

    def calculate_pI(sequence):
        """
        Calculate isoelectric point (pI) using the bisection method
        pI is the pH where net charge is zero
        """
        def charge_at_pH(pH, sequence):
            charge = 0.0
            for aa in sequence:
                # N-terminus and C-terminus
                if aa == sequence[0]:  # N-terminus
                    pKa = AA_DATA[aa]['pKa']
                    charge += 1 / (1 + 10 ** (pH - pKa))
                if aa == sequence[-1]:  # C-terminus
                    pKb = AA_DATA[aa]['pKb']
                    charge -= 1 / (1 + 10 ** (pKb - pH))
                
                # Charged side chains
                if AA_DATA[aa]['pKx'] is not None:
                    if AA_DATA[aa]['class'] in ['positive']:  # Basic (H, K, R)
                        pKx = AA_DATA[aa]['pKx']
                        charge += 1 / (1 + 10 ** (pH - pKx))
                    elif AA_DATA[aa]['class'] in ['negative']:  # Acidic (D, E)
                        pKx = AA_DATA[aa]['pKx']
                        charge -= 1 / (1 + 10 ** (pKx - pH))
            return charge

        # Bisection method to find pH where charge = 0
        low_pH = 0.0
        high_pH = 14.0
        tolerance = 0.01
        
        while high_pH - low_pH > tolerance:
            mid_pH = (low_pH + high_pH) / 2
            charge = charge_at_pH(mid_pH, sequence)
            if charge > 0:
                low_pH = mid_pH
            else:
                high_pH = mid_pH
        
        return round((low_pH + high_pH) / 2, 2)

    pI=calculate_pI(sequence)
        

    return {
        'sequence': sequence,
        'length': length,
        'molecular_weight': molecular_weight,
        'amino_acid_counts': aa_counts,
        'amino_acid_composition': composition,
        'average_hydropathy': avg_hydropathy,
        'net_charge_at_pH7': net_charge,
        'isoelectric_point': pI,
        'class_counts': class_counts
    }

# Streamlit app interface
st.title("FoldNFacts")
st.write("Your gateway to understand protein Structure and its Properties!!")

st.sidebar.title("Menu")
main_page = st.sidebar.radio("Go to", ["🏠 Home", "🛠️ Tools", "ℹ️ About", "👥 Contact"])

if main_page == "🏠 Home":
    st.header("👋 Welcome to FoldNFact!")
    st.markdown("""

    **FoldNFact** is an integrated webserver designed to help researchers, students, and professionals analyze protein sequences by providing insights into their **physicochemical properties** and **structural features**.

    With an intuitive interface and powerful tools, FoldNFact allows you to:

    - 🧪 **Analyze the physicochemical properties** of proteins (like molecular weight, isoelectric point, hydrophobicity, etc.)
    - 🧬 **Predict secondary structure elements** (alpha helices, beta sheets, and coils)
    - 🧩 **Visualize tertiary structures** (3D models)
    - 🔍 **Pattern Search** with built-in alignment tools

    ---

    ### 🚀 How to Get Started

    You can begin by selecting a tool from the **sidebar on the left**.  
    Each tool has a dedicated input form and clear instructions.  
    Simply **upload or paste your protein sequence** to get started.

    ---

    We’ve built **FoldNFact** to be **fast**, **reliable**, and **easy to navigate** — whether you're exploring a single protein or analyzing entire datasets.
    """)

elif main_page == "🛠️ Tools":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tools")
    home_subpage = st.sidebar.radio(
        "Select Tool", 
        ["🧪 Physiochemical Properties", "🔍 Motif Finder", "🧬 Structure"],
        key="home_tabs"
    )

    if home_subpage == "🧪 Physiochemical Properties":
        st.header("Protein Physio-Chemical Properties Calculator")
        sequence = st.text_area("Protein Sequence. Example:MKTIIALSYIFCLVFADYKDDDDK (Enter sequence without header and line break)", "")
        if st.button("Analyze"):
            try:
                results = calculate_protein_properties(sequence)
                st.subheader("Basic Information")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sequence Length", results['length'])
                col2.metric("Molecular Weight (Da)", f"{results['molecular_weight']:.2f}")
                col3.metric("Average Hydropathy", f"{results['average_hydropathy']:.2f}")
                col4, col5, _ = st.columns(3)
                col4.metric("Net Charge at pH 7", results['net_charge_at_pH7'])
                col5.metric("Isoelectric Point (pI)", results['isoelectric_point'])

                st.subheader("Amino Acid Composition")
                composition_df = pd.DataFrame.from_dict(results['amino_acid_composition'], orient='index', columns=['Percentage'])
                composition_df['Amino Acid'] = composition_df.index.map(lambda x: AA_DATA[x]['name'])
                composition_df = composition_df[['Amino Acid', 'Percentage']].sort_values('Percentage', ascending=False)
                st.dataframe(composition_df.style.format({'Percentage': '{:.2f}%'}),use_container_width=True)

                def download_csv(dataframe, filename="amino_acid_composition.csv"):
                    csv = dataframe.to_csv(index=False)
                    return csv.encode()
                
                # Provide CSV download button
                csv_data = download_csv(composition_df)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="amino_acid_composition.csv",
                    mime="text/csv"
                )

                fig, ax = plt.subplots(figsize=(8,4))  # Taller since it's horizontal
                palette = sns.color_palette("Set3", len(composition_df))
                composition_df.plot.barh(y='Percentage', x='Amino Acid', ax=ax, legend=False, color=palette)
                plt.xlabel("Percentage (%)")
                plt.title("Amino Acid Composition", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

                def download_image(fig, filename="amino_acid_composition.png"):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    return buf.getvalue()

                # Provide Image download button
                img_data = download_image(fig)
                st.download_button(
                    label="🖼️ Download Chart as Image",
                    data=img_data,
                    file_name="amino_acid_composition.png",
                    mime="image/png"
                )

                st.subheader("Amino Acid Class Distribution")
                class_df = pd.DataFrame.from_dict(results['class_counts'], orient='index', columns=['Count'])
                class_df['Percentage'] = class_df['Count'] / results['length'] * 100
                class_df = class_df.sort_values('Count', ascending=False)
                st.dataframe(class_df.style.format({'Percentage': '{:.2f}%'}))

                csv_buffer = io.StringIO()
                class_df.to_csv(csv_buffer)
                st.download_button(
                    label="📥 Download Class Distribution as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="amino_acid_class_distribution.csv",
                    mime="text/csv"
                )

                fig2, ax2 = plt.subplots(figsize=(7,4))
                palette = sns.color_palette("pastel", len(class_df))
                wedges, texts, autotexts = ax2.pie(
                    class_df['Count'],
                    autopct='%1.1f%%',
                    startangle=140,
                    colors=palette,
                    radius=0.5,  # Smaller radius
                    textprops={'fontsize': 10},
                    pctdistance=0.8,
                    labeldistance=1.0  # Push labels out a bit
                )

                # Set title with spacing
                ax2.set_title("Amino Acid Class Distribution", fontsize=12, pad=15)
                ax2.axis('equal')  # Keeps it circular

                # Add legend instead of pie labels (optional)
                ax2.legend(wedges, class_df.index, title="Classes", loc="center left", bbox_to_anchor=(0.95,0.5))

                plt.tight_layout()
                st.pyplot(fig2)

                # Download PNG of the chart
                png_buffer = io.BytesIO()
                fig2.savefig(png_buffer, format="png", bbox_inches='tight')
                st.download_button(
                    label="🖼️ Download Pie Chart as PNG",
                    data=png_buffer.getvalue(),
                    file_name="class_distribution_pie_chart.png",
                    mime="image/png"
                )

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif home_subpage == "🔍 Motif Finder":
        st.header("🔍 Protein Motif Finder")
        col1, col2 = st.columns(2)
        with col1:
            sequence = st.text_area("Protein Sequence. Example:MKTIIALSYIFCLVFADYKDDDDK (Enter sequence without header and line break)","", height=200).upper().strip()
        with col2:
            motif_pattern = st.text_input("Motif Pattern (regex)", "G..G")
            st.markdown("**Common Motif Examples:**")
            st.code("""N-glycosylation: N[^P][ST]
O-glycosylation: [ST]A
Phosphorylation: [ST]
Zinc finger: C..H""")

        if st.button("Find Motifs"):
            if not sequence or not motif_pattern:
                st.warning("Please enter both sequence and motif pattern")
            else:
                try:
                    re.compile(motif_pattern)
                    matches = []
                    for match in re.finditer(f"(?=({motif_pattern}))", sequence):
                        start = match.start(1)
                        end = match.end(1)
                        matches.append({
                            "Position": f"{start+1}-{end}",
                            "Sequence": sequence[start:end],
                            "Context": f"...{sequence[max(0,start-3):end+3]}..."
                        })
                    
                    if matches:
                        st.success(f"Found {len(matches)} motif(s)")
                        df = pd.DataFrame(matches)
                        st.dataframe(df.style.highlight_max(axis=0))

                        st.subheader("Sequence Map")
                        seq_display = list(sequence)
                        for match in matches:
                            start = int(match['Position'].split('-')[0]) - 1
                            end = int(match['Position'].split('-')[1])
                            for i in range(start, end):
                                seq_display[i] = f"<span style='color:red;font-weight:bold'>{seq_display[i]}</span>"
                        st.markdown("".join(seq_display), unsafe_allow_html=True)

                        st.subheader("Position Frequency")
                        pos_counts = defaultdict(int)
                        for match in matches:
                            pos_counts[match['Sequence']] += 1
                        fig, ax = plt.subplots()
                        pd.Series(pos_counts).plot.bar(ax=ax)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    else:
                        st.warning("No motifs found in the sequence")
                except re.error:
                    st.error("Invalid regular expression pattern")

    elif home_subpage == "🧬 Structure":
        st.header("Protein Visualization")
        st.write("Protein Structure through ESM Fold")

        def clean_sequence(seq):
            lines = seq.splitlines()
            cleaned = ''.join(line.strip().replace(" ", "") for line in lines if not line.startswith('>'))
            return cleaned.upper()

        txt = st.text_area("Input Sequence (Enter sequence without header and line break)", "", height=175)
        sequence = clean_sequence(txt)
        

        #stmol
        def render_mol(pdb):
            pdbview = py3Dmol.view()
            pdbview.addModel(pdb,'pdb')
            pdbview.setStyle({'cartoon':{'color':'spectrum'}})
            pdbview.setBackgroundColor('white')#('0xeeeeee')
            pdbview.zoomTo()
            pdbview.zoom(2, 800)
            #pdbview.spin(True)
            showmol(pdbview, height = 500,width=800)
        
        #ESM Fold
        def update(sequence):
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=sequence)

            if response.status_code == 200:
                pdb_string = response.text

                # Save PDB file
                with open('predicted.pdb', 'w') as f:
                    f.write(pdb_string)

                # Load structure and compute plDDT
                struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
                b_value = round(struct.b_factor.mean(), 4)

                # Display visualization and info
                st.subheader('🧬 Predicted Protein Structure')
                render_mol(pdb_string)

                st.subheader('📈 plDDT Score')
                st.write('plDDT is a per-residue confidence estimate (0–100).')
                st.info(f'Average plDDT: **{b_value}**')

                st.download_button(
                    label="📥 Download PDB",
                    data=pdb_string,
                    file_name='predicted.pdb',
                    mime='text/plain'
                )
            else:
                st.error("❌ Structure prediction failed. Try again later.")

        # --- Predict button and output placeholder ---
        predict_button = st.button("🔍 Predict Structure")
        output_placeholder = st.empty()

        if predict_button:
            if not sequence or not sequence.isalpha():
                st.warning("⚠️ Please enter a valid amino acid sequence (letters only).")
            else:
                with output_placeholder.container():
                    with st.spinner("⏳ Predicting structure using ESMFold..."):
                        update(sequence)

    elif home_subpage == "📊 Visualize":
        st.header("Protein Visualization")
        st.write("Coming soon!")

elif main_page == "ℹ️ About":
    st.header("About this Webserver")
    st.markdown("""
    ### 🧭 Overview

    **FoldNFact** is an all-in-one webserver for analyzing protein sequences and structures.  
    It combines physicochemical property calculators, secondary and tertiary structure predictors,  
    and interactive visualization tools. Designed for ease of use, FoldNFact supports fast,  
    reliable insights for researchers, students, and professionals in protein science.
    """)

    # Initialize session state
    if "about_section" not in st.session_state:
        st.session_state.about_section = None

    # Arrange buttons in a row using columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Objective"):
            st.session_state.about_section = "objective"

    with col2:
        if st.button("Features"):
            st.session_state.about_section = "features"

    with col3:
        if st.button("How to use"):
            st.session_state.about_section = "howto"

    with col4:
        if st.button("Future Directions"):
            st.session_state.about_section = "future"

    # Display content based on the button clicked
    section = st.session_state.about_section

    if section == "objective":
        st.subheader("🎯 Objective")
        st.markdown("""
        The objective of this webserver is to provide an easy-to-use platform for analyzing protein sequences, predicting their structures, and understanding their physicochemical properties. 
        Key goals include:
        - Analyzing protein sequence properties
        - Predicting secondary and tertiary structures
        - Visualizing results interactively
        """)
    elif section == "features":
        st.subheader("🔧 Features")
        st.markdown("""
        - **Protein sequence input and cleaning**: Easy Prediction of the known and unknown Protein sequence.
        - **Structural prediction**: Predict secondary and tertiary structures of the known and unknown Protein sequence using tools like ESMFold.
        - **3D Visualization**: Visualize predicted structures with confidence scores (plDDT).
        """)
    elif section == "howto":
        st.subheader("📘 How to Use")
        st.markdown("""
        - **Step 1**: Select a tool of choice from the sidebar.
        - **Step 2**: Input your protein sequence in the Input box.
        - **Step 3**: Click 'Predict' to analyze the sequence and view results.
        - **Step 4**: View and download the results, including predicted structures and physicochemical properties.
        """)
    elif section == "future":
        st.subheader("🔮 Future Directions")
        st.write("""
        The future development of FoldNFact includes:
        - **AlphaFold integration**: Implementing the highly accurate AlphaFold for structure prediction.
        - **Input Sequence**:Easier input of the sequence
        - **Motif/Domain prediction**: Adding tools for motif and domain identification.
        - **Functional annotation**: Adding functionality to annotate sequences with functional data.
        - **Enhanced visualization**: Adding support for more advanced 3D structure visualization.
        """)

elif main_page == "👥 Contact":
    st.title("Our Team")
    # Creator Section
    st.markdown("""
    <div class="Creator-section">
        <h3>👩‍🔬 About the Creator</h3>
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://media.licdn.com/dms/image/v2/D4D03AQGv0fpgreIM3g/profile-displayphoto-shrink_400_400/B4DZVQE3fmGkAg-/0/1740805209414?e=1752105600&v=beta&t=qoNcQowPg06w-oWwnKaQx6OlU0DBYy1yD1VfS6_zKTo"
                 width="120" style="border-radius: 50%; border: 3px solid #6A5ACD;">
            <div class="about-section">
                <h4>Jyoti Rana</h4>
                <p>M.Sc. Bioinformatics Student, DES Pune University</p>
                <p>Jyoti Rana is a postgraduate student specializing in Bioinformatics at DES Pune University. This web server was developed as part of her academic project, focusing on the analysis of protein physicochemical properties and the prediction of protein structures through integrated computational tools.</p>
                <p>She has a strong interest in protein science and structural bioinformatics, and is continually exploring ways to apply computational methods to solve biological problems. Jyoti is passionate about learning and expanding her knowledge in the field of bioinformatics, with the aim of contributing to impactful scientific research.</p>
                <p>Email: <a href="mailto:jyotirana1202@gmail.com">jyotirana1202@gmail.com</a></p> <!-- Added email for Creator -->
            </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
   
    # Mentorship Section
    st.markdown("""
    <div class="mentorship-section">
        <h3>👨‍🏫 Mentorship</h3>
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://media.licdn.com/dms/image/v2/D5603AQF9gsU7YBjWVg/profile-displayphoto-shrink_400_400/B56ZZI.WrdH0Ag-/0/1744981029051?e=1752105600&v=beta&t=F4QBDSEgjUvnBS00xPkKqPTLI0jQaMpYefaOzARY1Yg"
                 width="120" style="border-radius: 50%; border: 3px solid #6A5ACD;">
            <div>
                <h4>Dr. Kushagra Kashyap</h4>
                <p>Assistant Professor (Bioinformatics), Department of Life Sciences, School of Science and Mathematics, DES Pune University</p>
                <p>This project was developed under the guidance of Dr. Kashyap, who provided valuable insights and mentorship
                throughout the development process. His expertise in bioinformatics and computational biology was instrumental
                in shaping this project.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)