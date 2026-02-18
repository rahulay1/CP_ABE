import os
import requests

# The complete list of URLs provided (duplicates removed)
image_urls = [
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_cce34b7862f645a1bc233e8fb36a445f.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_fa03187e49814d8f9101e6e9a15e8f17.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_1eefa1da3f9d43dba974e52777655452.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_9303d92ffa1f4c6ca4239795a3b82c03.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_4630f10eb5c94b0e8b8b248642023128.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_f8d708807c5840fb958c83572e396c8d.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_9755d2326e7f4151a62585a7bd93f0a2.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_2f4dce4d50794adca766738c2c099db5.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_a46929bcc66e4950ab00c8d6852a95f6.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_aeb578eb320d42aab76dfe09d98d8e52.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_ba07cff024d14923aa872399e86ed0ce.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_7ca70fc34e11470f9e23090ecf579082.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_5bcd125d269a45f8a25e4aac257400fb.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_d9186a614551495b99300f72b4dd4911.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_0a3beecfa5304826b0b92908044f6dec.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_e140da3da9254b088b4c3396668c0ee9.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_50e8d462c66340d686a4a300673eaa35.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_7c3807870b1044b7a3c25e4f57a88c82.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_107bcbfda5364b65960983c6cc68a949.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_803b6d5d92404b4786029571c2a3dcbf.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_f0a20255cc564f07ae8cc9443433d131.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_3dd1b6b8af284927950779d9ce2ff5c8.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_36f0001df72b4998b54015022e9b50f7.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_3c2654c359d34c80b01ce4f3ff798c4f.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_49ec198a67404147b6195f7f970f85b0.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_0adbb6f7ba044b1f9d455361ce551328.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_e5f0fa54e5514befaa5987af4e0a6451.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_447ed4dac9aa4fa58e82f008f37b7470.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_2801ef8cecc5449985b0988658b6d87d.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_d6b0d90a8c03441b8e2206226a4ed1d1.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_0556d9ba25fa4904b7ffc86a30551309.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_87d012abfbda4b2da297b366753fe9b9.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_08e1739ca26a47288f4b19954e10d3c2.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_40832445eeea45e9806c15e9ab2145da.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_4487e80b533842e0858439f664607364.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_079aa8cd350e45a5844d1d9cbd9fbacb.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_802142b88d544f5ba9509b9c020eb018.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_8715f6d651de4e08bad68ab647ccd7e6.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_8e49ca9ea7f7422185412d18578103a1.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_a572049d79164f6b9cc7b6d5a3caa775.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_5aff5293d84b491ab0a595f0b1bb0ff2.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_4e009ca747654feaafcf771479edc574.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_a6140838473f4cd19d92ad503cc6ed92.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_30e72d00137442d0b581351ec9baa87f.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_e0ae86b5f4a449279ed640c5ee40a6c0.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_bd4ef5e8509c441daef33032bf013d83.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_fddc6ebd771a4ce5809c33e4d09cd61d.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_605c78be8957402e867005a9a7037fb0.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_f2df559370f54495bae6690c8c4df353.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_ff034d65326b4ac2b9c418e9e6a5662d.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_885f3ca3222b4f2e968e502c3411c5fc.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_12e3cff41a0b46efbad5acd2e8284539.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_d939e80ec8ac48a7888f0777f2ba7613.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_f45cce35cd4d48e38294e7026bf3a4d9.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_65e9587ffd10418c9834499553d02bf4.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_f5d01a74863d44f48cbe5b7ea4d987cc.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_f540b870270d423886c92be42f172689.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_ade959ba5bae4d5da1e152f095ab127b.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_ef9ba076199146f3b04131f20a513abe.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_52aff303f5fc4993a8e742ab1f87843d.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_2aed017cb0fa44efa7bf824a8645f8dc.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_2f6205723414479d8ef6e30809e08271.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_6f9a068b9ad6488683adee14d206806d.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_6e10b5ee41ca474ea4db8c9c6dff3a79.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_f7967140183e41c5b496fd1d29ad11f9.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_6bdb16e19fde42c6b09ec68781dc6294.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_c481d73ff0064de88c6dd86026af3d26.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_baf45b46233c4c979939cd5123c0b207.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_31e0c79ec29146278c51df440c4e8754.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_554cd82bc9734f9284d15714d19f6c52.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_3d68b734010842779f8ab22e8c081016.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_a54cd4b142b9439c9cdf11b993a1614e.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_5b75d6d633b943dc9ad576ae98c654e3.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_dea986fd11f148bb98434f515680d6fa.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_a5ac06971ad844178ad325370a28beef.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_c4b454de6d8e40fd86f4ecfda4f4bc6e.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_6aecd57c14734b5d97b6660007a832b3.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_32a0a2ba544945909b94aafa7d4a662b.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_b586de6fd7234f81a49f3324370df09f.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_7e364ab8a3654854b92ce21c6e90464a.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_8d6a0ca623b74740ae79bd51c5b6bfaf.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_79401bbcb5514a769282a75def7bd4fe.jpg",
"https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_98a24ba726e44d18943bacaa6e44ae88.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_1ace0b6e7e8a41e7bb1a2743ce1cde20.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_b20c5dd9e8754cbba25f1245a4c7f2f9.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_1a5ab5d0dd284bdda84587509c4ddba3.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_b9c6c3197f0c42c6aa66e6725c01ee93.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_b6315ebfc85d4acb8b193abb3931d62d.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_996f0aa8ff1a4969b26f4ef153fd5047.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_5d0656f56e3240b3a12990c421406ba8.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_67c0032193f044c8ad706c0c731433b6.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_29b21c50255d41089b1ac4e1c3f545e2.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_3c384a0a48654028b53eccad7af4fdac.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_74cf6035245243fa92f45f84482588fc.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_898295f1a93d447da9d564e0b234d459.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_fa7084562df34fc7858d371645912ec0.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_7a8def4793d644ddb19824740be640cb.jpg",
    "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_758802a2e3bd40138b3fd919986f243e.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_81dbceefcff345a3bc96895c7545c0b8.jpg",
"https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_9521811c8e2b4c9ab73e3a13cb4359fd.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_265cd68c08024e9f9e43eff11e344972.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_9ad432c1af5846519487dc20e67e3cfa.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_9e44182559bb486898bd0d7e019f37b3.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_0d105ed3edcc411ca2ee24c6a9a5d021.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_918d7009b23140bfbf59460ad072acbd.jpg",
 "https://listfast-uploader.s3.eu-north-1.amazonaws.com/images/single_76fad1d0421d4ed4994a95630ea23ba1.jpg"
]


def download_images(urls, folder="downloaded_images"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    print(f"Starting download of {len(urls)} images...")

    for i, url in enumerate(urls):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # Extract filename from URL
                filename = url.split("/")[-1]
                filepath = os.path.join(folder, filename)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"[{i + 1}/{len(urls)}] Downloaded: {filename}")
            else:
                print(f"[{i + 1}/{len(urls)}] Failed to download {url}: Status {response.status_code}")
        except Exception as e:
            print(f"[{i + 1}/{len(urls)}] Error downloading {url}: {e}")

    print("\nDownload complete!")


if __name__ == "__main__":
    download_images(image_urls)