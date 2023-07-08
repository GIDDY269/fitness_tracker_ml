import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl



#load data
df = pd.read_pickle('../../artifacts/transformed_data.pkl')

set_df = df[df['set'] == 1]
plt.plot(set_df['acc_y']) # duration of the set

plt.plot(set_df['acc_y'].reset_index(drop=True))
plt.xlabel('number of samples')

# adjusting settings

mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = [20,5]
mpl.rcParams['figure.dpi'] = 100


# plot for all the exercises
for label in df['label'].unique() :
    subset = df[df['label'] == label ]
    fig, ax = plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop=True),label=label)
    plt.legend() ;

# plot fraction of the data
for label in df['label'].unique() :
    subset = df[df['label'] == label ]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True),label=label)
    plt.legend() ;


# comparing medium and heavy set

category_df = df.query("label == 'squat'").query("participants == 'A' ").reset_index()
category_df.groupby('category')['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend();

category_df = df.query("label == 'bench'").query("participants == 'B' ").reset_index()
category_df.groupby('category')['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend();

category_df = df.query("label == 'squat'").query("participants == 'A' ").reset_index()
category_df.groupby('category')['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('acc_x')
plt.legend();

category_df = df.query("label == 'ohp'").query("participants == 'A' ").reset_index()
category_df.groupby('category')['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend();


# comparing participants

participant_df = df.query("label == 'bench' ").sort_values('participants').reset_index()
participant_df.groupby('participants')['acc_y'].plot()
plt.title('participants bench press',fontweight='bold')
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend();


participant_df = df.query("label == 'squat' ").sort_values('participants').reset_index()
participant_df.groupby('participants')['acc_y'].plot()
plt.title('participants squat press',fontweight='bold')
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend();

participant_df = df.query("label == 'ohp' ").sort_values('participants').reset_index()
participant_df.groupby('participants')['acc_y'].plot()
plt.title('participants over head press',fontweight='bold')
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend();

#plot multiple componets

labels = df['label'].unique()
participants = df['participants'].unique()

for label in labels:
    for participant in participants :
        all_axis_df = df.query(f"label == '{label}'"
                               ).query(
                                   f"participants == '{participant}'"
                                       ).reset_index()
        if len(all_axis_df) > 0 :
            all_axis_df[['acc_x','acc_y','acc_z']].plot()
            plt.xlabel('acc measure')
            plt.ylabel('samples')
            plt.title(f'{label} {participant}')
            plt.legend()

for label in labels:
    for participant in participants :
        all_axis_df = df.query(f"label == '{label}'"
                               ).query(
                                   f"participants == '{participant}'"
                                       ).reset_index()
        if len(all_axis_df) > 0 :
            all_axis_df[['gyr_x','gyr_y','gyr_z']].plot()
            plt.xlabel('acc measure')
            plt.ylabel('samples')
            plt.title(f'{label} {participant}')
            plt.legend()


# combining accelerometer and gyroscope plots
labels = df['label'].unique()
participants = df['participants'].unique()

for label in labels:
    for participant in participants :
        combined_df = df.query(f"label == '{label}'"
                               ).query(
                                   f"participants == '{participant}'"
                                       ).reset_index()
        if len(combined_df) > 0 :

            fig, ax = plt.subplots(nrows=2,sharex=True,figsize=[20,10])
            combined_df[['acc_x','acc_y','acc_z']].plot(ax = ax[0])
            combined_df[['gyr_x','gyr_y','gyr_z']].plot(ax = ax[1])

            ax[0].legend(
                loc = 'upper center',bbox_to_anchor= [0.5,1.15], fancybox = True,shadow = True
            )

            ax[1].legend(
                loc = 'upper center',bbox_to_anchor= [0.5,1.15], fancybox = True,shadow = True
            )

            ax[1].set_xlabel('samples')
            os.makedirs('../../reports',exist_ok=True)
            plt.savefig(f'../../reports/{label.title()} ({participant}).png')









            

        




